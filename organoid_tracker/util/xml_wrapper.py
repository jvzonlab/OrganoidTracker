"""An XML wrapper for easier parsing. It allows you to safely write code like this:

    metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"].value_float()

If any of the elements in the chain doesn't exist, you will just end up with None at the end, instead of crashing
halfway through.
"""
from typing import Optional, Iterator
import xml.dom.minidom
from xml.dom.minidom import Element


class XmlWrapper:
    """An XML wrapper for easier parsing. It allows you to safely write code like this:

        metadata["ImageDocument"]["Metadata"]["Scaling"]["Items"]["Distance"].value_float()

    One limitation is that you can't extract text from elements that have both text and subelements.
    """

    _element: Optional[Element]

    def __init__(self, element: Optional[Element]):
        self._element = element

    def is_none(self) -> bool:
        """Checks if this element is equal to NONE_ELEMENT."""
        return self._element is None

    def __getitem__(self, item: str) -> "XmlWrapper":
        """Returns the first child element with the given name. If it doesn't exist, returns NONE_ELEMENT."""
        if self._element is None:
            return NONE_ELEMENT

        elements = self._element.getElementsByTagName(item)
        if len(elements) > 0:
            return XmlWrapper(elements[0])
        return NONE_ELEMENT

    def value_str(self) -> Optional[str]:
        """If this tag has just one text node as a child, returns its value. Otherwise, returns None."""
        if self._element is None:
            return None
        if len(self._element.childNodes) == 1 and self._element.firstChild.nodeType == self._element.TEXT_NODE:
            return self._element.firstChild.nodeValue
        return None

    def value_float(self) -> Optional[float]:
        """If this tag has just one text node as a child, returns its value as a float. If the value is not a valid
        float, returns None."""
        value = self.value_str()
        if value is not None:
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def attr_str(self, name: str) -> Optional[str]:
        """Returns the value of the attribute with the given name. If the attribute doesn't exist, returns None."""
        if self._element is None:
            return None
        return self._element.getAttribute(name)

    def attr_float(self, name: str) -> Optional[float]:
        """Same as attr_str, but returned as a float. Returns None if the attribute was not found, or couldn't be
        parsed as a float."""
        value = self.attr_str(name)
        if value is not None:
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def __iter__(self) -> Iterator["XmlWrapper"]:
        """Iterates over all subelements. Skips text nodes."""
        if self._element is None:
            return iter([])
        return (XmlWrapper(node) for node in self._element.childNodes if node.nodeType == self._element.ELEMENT_NODE)

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        if self._element is None:
            return "XmlWrapper(None)"
        return self._element.toprettyxml().strip()


# If an XML element doesn't exist, this is returned instead of None, so that your code doesn't need to check for None
# all the time.
NONE_ELEMENT = XmlWrapper(None)


def read_xml(xml_string: str) -> XmlWrapper:
    return XmlWrapper(xml.dom.minidom.parseString(xml_string).documentElement)
