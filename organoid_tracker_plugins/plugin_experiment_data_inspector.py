import collections.abc
import html
import inspect
import types
from typing import Optional, Dict, Any, Iterable, NamedTuple, List

from organoid_tracker.core import TimePoint, UserError
from organoid_tracker.core.position import Position
from organoid_tracker.gui import dialog
from organoid_tracker.gui.window import Window
from organoid_tracker.text_popup.text_popup import RichTextPopup

_PARAM_BOUNDARY = "*---*"

_MAX_SUMMARY_STRING_LENGTH = 100


def get_menu_items(window: Window) -> Dict[str, Any]:
    return {
        "Help//Basic-Show data inspector...": lambda: _show_inspector(window)
    }


class _Element(NamedTuple):
    parent_name: str
    parent_object: Any
    element_name: str

    @property
    def signature(self) -> str:
        attribute = getattr(self.parent_object, self.element_name)
        if callable(attribute):
            # It's a method
            signature = str(inspect.signature(attribute))
            if " ->" in signature:  # Chop off type, not used in the display name
                signature = signature[:signature.index(" ->")]
            return signature
        return ""

    @property
    def link(self):
        attribute = getattr(self.parent_object, self.element_name)
        if callable(attribute):
            # It's a method
            signature = inspect.signature(attribute)
            if len(signature.parameters) == 0:
                return "goto:" + self.parent_name + "." + self.element_name + "()"

            # We format function calls like `test_method(*---*a*---*b*---*)` instead of `test_method(a,b)`
            # so that we can later easily split parameters
            # Otherwise parsing for example test_method(par1: Union[str, float], par2: str) would be difficult
            link = "goto:" + self.parent_name + "." + self.element_name + "(*---*"
            signature.__str__()
            for parameter in signature.parameters.values():
                annotation = inspect.formatannotation(parameter.annotation)\
                    if parameter.annotation != parameter.empty else "typing.Any"
                link += parameter.name + ": " + annotation + "*---*"
            return link + ")"
        # It's a field
        return "goto:" + self.parent_name + "." + self.element_name

    def is_clickable(self) -> bool:
        attribute = getattr(self.parent_object, self.element_name)

        if callable(attribute):
            # It's a method
            signature = inspect.signature(attribute)
            return signature.return_annotation != signature.empty
        # It's a field
        return True

    @property
    def comment(self) -> Optional[str]:
        try:
            # Get attribute on class first (to get docs of @property-objects)
            attribute = getattr(self.parent_object.__class__, self.element_name)
        except AttributeError:
            # Else, try instance property
            attribute = getattr(self.parent_object, self.element_name)
        comment = attribute.__doc__
        if comment is not None and ") ->" in comment:
            return None  # Ignore this comment, it likely comes from base python
        return comment

    def type_str(self) -> Optional[str]:
        """Gets the type of the returned element."""
        attribute = getattr(self.parent_object, self.element_name)
        if callable(attribute):
            # It is a method
            signature = inspect.signature(attribute)
            if signature.return_annotation != signature.empty:
                return _format_type(str(signature.return_annotation))

        # It is a field
        if hasattr(self.parent_object, "__annotations__") and self.element_name in self.parent_object.__annotations__:
            element_type = str(self.parent_object.__annotations__[self.element_name])
            return _format_type(element_type)

        return None

    def to_html(self) -> str:
        if self.is_clickable():
            text = "<code><a href=\"" + self.link + "\">" + self.element_name + "</a>"
        else:
            text = "<code><b>" + self.element_name + "</b>"
        text += self.signature

        type_str = self.type_str()
        if type_str is not None:
            text += ": " + html.escape(type_str)
        text += "</code>"
        if self.comment is not None:
            text += "<br><i><small>" + html.escape(self.comment) + "</small></i>"
        return text


def _get_summary(element: Any) -> str:
    output = "<p>Class: <code>" + html.escape(_format_type(str(type(element)))) + "</code>"

    element_str = str(element)
    if not element_str.startswith("<"):
        if len(element_str) > _MAX_SUMMARY_STRING_LENGTH:
            element_str = element_str[:30] + " ... " + element_str[-30:]
        output += "<br>Value: <code>" + html.escape(element_str) + "</code>"
    output += "</p>"

    return output


def _prompt_single_parameter(window: Window, name: str, type: str) -> Optional[str]:
    """Asks for a value of a parameter"""
    # Find appropriate default value
    default_value = ""
    if "TimePoint" in type:
        current_time_point: TimePoint = window.display_settings.time_point
        default_value = repr(current_time_point)
    elif "Position" in type:
        default_value = repr(Position(0, 0, 0,
                                      time_point_number=window.display_settings.time_point.time_point_number()))
    elif "ImageChannel" in type:
        try:
            image_channel_index = window.get_experiment().images.get_channels()\
                .index(window.display_settings.image_channel)
        except IndexError:
            image_channel_index = 0
        default_value = "experiment.images.get_channels()[" + str(image_channel_index) + "]"
    elif "str" in type:
        default_value = "\"some value\""
    elif "NoneType" in type:
        default_value = "None"
    elif "int" in type:
        default_value = "0"
    elif "bool" in type:
        default_value = "False"
    value_str = dialog.prompt_str(f"Parameter \"{name}\"",
                                  f"Please provide a value for the parameter \"{name}\" (type: {type})",
                                  default=default_value)
    return value_str


class _DataInspectorPopup(RichTextPopup):
    _window: Window

    _navigation_stack: List[str]

    def __init__(self, window: Window):
        self._window = window
        self._navigation_stack = list()

    def get_title(self) -> str:
        return "Inspector"

    def _get_inspector(self, parent_element_name: str) -> Optional[str]:
        if not parent_element_name.startswith("experiment"):
            return None  # Cannot handle this

        experiment = self._window.get_experiment()
        parent_element_name = self._fill_in_parameters(parent_element_name)
        if parent_element_name is None:
            return None
        parent_element = eval(parent_element_name, globals(), {"experiment": experiment})

        self._navigation_stack.append(parent_element_name)

        output = f"<h2>Contents of <code>{html.escape(parent_element_name)}</code>:</h2>"
        output += self._get_back_link()
        output += _get_summary(parent_element)
        output += _get_detailed_contents(parent_element_name, parent_element)

        return output

    def navigate(self, url: str) -> Optional[str]:
        if url == self.INDEX:
            return self._get_inspector("experiment")
        if url == "goto-previous:":
            if len(self._navigation_stack) < 2:
                return None
            self._navigation_stack.pop()  # Removes current page
            return self._get_inspector(self._navigation_stack.pop())  # Goes to previous
        if url.startswith("goto:"):
            parent_element_name = url[len("goto:"):]
            return self._get_inspector(parent_element_name)
        return None

    def _get_back_link(self) -> str:
        if len(self._navigation_stack) > 1:
            parent_name = self._navigation_stack[-2]
            return "<p><a href=\"goto-previous:\">← Back to <code>" + parent_name + "</code></a>"
        return ""

    def _fill_in_parameters(self, eval_code: str) -> Optional[str]:
        """Fills in the parameters in a call like test_method(*---*a*---*b*---*) """
        if _PARAM_BOUNDARY not in eval_code:
            return eval_code  # No paramaters

        params_start = eval_code.index(_PARAM_BOUNDARY)
        params_end = eval_code.rindex(_PARAM_BOUNDARY)
        if params_start == params_end:
            return None  # Invalid string

        params_str = eval_code[params_start + len(_PARAM_BOUNDARY):params_end]
        filled_in = []
        for param_str in params_str.split(_PARAM_BOUNDARY):
            name, type = param_str.split(":", maxsplit=2)
            name = name.strip()
            type = type.strip()
            value_str = _prompt_single_parameter(self._window, name, type)
            if value_str is None:
                return None
            if value_str.endswith(","):
                raise UserError("Invalid value", f"Could not compile \"{value_str}\"\n\n"
                                                 f"Expression may not end with a comma")
            try:
                compile(value_str, name, "eval")
            except Exception as e:
                raise UserError("Invalid value", f"Could not compile \"{value_str}\"\n\n{e}")
            filled_in.append(value_str)
        return eval_code[:params_start] + ", ".join(filled_in) + eval_code[params_end + len(_PARAM_BOUNDARY)]


def _format_type(type_name: str) -> str:
    if type_name.startswith("<class '"):
        # Strip off "<class 'our.class.name'>"
        type_name = type_name[len("<class '"): -2]

    type_name = type_name.replace("NoneType", "None")

    return type_name


def _get_detailed_contents(parent_name: str, obj: Any) -> str:
    """Gives a nice HTML overview of all contents of the instance, with clickable links."""
    if "organoid_tracker." in str(type(obj)):
        output = "<ul>"
        for element in _get_fields_and_methods_of_organoid_tracker_instance(parent_name, obj):
            output += "<li>" + element.to_html() + "</li>"
        output += "</ul>"
        return output
    elif isinstance(obj, str):
        if len(obj) > _MAX_SUMMARY_STRING_LENGTH:
            # Was not fully displayed in summary, so display here
            return "<h2>Full value:</h2><p><code><small>" + html.escape(obj) + "</small></code></p>"
        return ""
    elif isinstance(obj, collections.abc.Iterable):
        return _get_detailed_contents_of_iterable(obj)
    else:
        return ""


def _get_detailed_contents_of_iterable(obj: Iterable[Any]):
    element_count = 0
    collection_type = _format_type(str(type(obj)))
    output = "<ul>"
    for element in obj:
        output += "<li><code>" + html.escape(str(element)) + "</code></li>"
        element_count += 1
    if element_count == 0:
        return "<p>Empty " + html.escape(collection_type) + ".</p>"
    output += "</ul>"
    output = "<p>" + html.escape(collection_type.title()) + " with <b>" \
             + str(element_count) + "</b> elements.</p>" + output
    return output


def _get_fields_and_methods_of_organoid_tracker_instance(parent_name: str, obj: Any) -> Iterable[_Element]:
    for name in dir(obj):
        if name.startswith("_"):
            continue  # Ignore private fields and methods

        attribute = getattr(obj, name)
        if isinstance(attribute, types.FunctionType):
            continue  # It's a static method (==function). ignore

        yield _Element(parent_name=parent_name, parent_object=obj, element_name=name)


def _show_inspector(window: Window):
    dialog.popup_rich_text(_DataInspectorPopup(window))
