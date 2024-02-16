import unittest

from organoid_tracker.util.xml_wrapper import read_xml


class TextXmlWrapper(unittest.TestCase):

    def test_text_extraction(self):
        """Tests if the text of a tag is extracted correctly."""
        xml = "<a><b><c>CONTENTS</c></b></a>"
        wrapper = read_xml(xml)
        self.assertEqual("CONTENTS", wrapper["b"]["c"].value_str())

    def test_no_text_extraction(self):
        xml = "<a><b><c>CONTENTS</c></b></a>"  # Note that there is no text in the "b" tag, but a "c" tag instead
        wrapper = read_xml(xml)
        self.assertIsNone(wrapper["b"].value_str())  # So the "b" tag shouldn't return any text

    def test_iteration(self):
        xml = ("<a><b>"
                   "<c>foo</c>"
                   "<c>bar</c>"
                   "<c>baz</c>"
               "</b></a>")

        xml_element = read_xml(xml)
        strings = [element.value_str() for element in xml_element["b"]]
        self.assertEqual(["foo", "bar", "baz"], strings)

    def test_no_iteration(self):
        xml = "<a><b><c>foo</c></b></a>"

        xml_element = read_xml(xml)
        sub_elements = list(xml_element["b"]["c"])
        self.assertEqual([], sub_elements)

    def test_float_extraction(self):
        xml = "<a><b><c>3.14</c></b></a>"
        wrapper = read_xml(xml)
        self.assertEqual(3.14, wrapper["b"]["c"].value_float())

    def test_read_attributes(self):
        document = read_xml("""<Metadata><Items>
                    <Distance Id="X">
                        <Value>3.7739355498356322e-007</Value>
                    </Distance>
                    <Distance Id="Y">
                        <Value>3.7739355498356322e-007</Value>
                    </Distance>
                    <Distance Id="Z">
                        <Value>3.0000000000000001e-006</Value>
                    </Distance>
                </Items></Metadata>""")
        distance_items = document["Items"]
        ids = [element.attr_str("Id") for element in distance_items]
        self.assertEqual(["X", "Y", "Z"], ids)
