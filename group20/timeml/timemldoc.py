from xml.dom import minidom as dom

class TimeMLDoc:
    """ Class that represents a single TimeML annotated file"""
    def __init__(self, path):
        """ Initializing variables """
        self.path = path
        self.root_ = dom.parse(self.path).documentElement
        # this is the root xml element that contains the sentences
        # of the document
        self.text_node = None
          
    def raw(self):
        """ Returns the raw text of the TimeML document """
        s = self.sentences(tag=False)
        self.text_ = ' '.join(s)
        return self.text_

    def __set_text_node(self, n):
        """ Recursively descends xml tree to find the 'TEXT' node"""
        if n.nodeType == n.ELEMENT_NODE and n.tagName == 'TEXT':
            self.text_node = n
        else:
            for child in n.childNodes:
                self.__set_text_node(child)
                
    def sentences(self, tag=False, tag_method=None):
        """ Returns the sentences tagged or untagged """
        self.__set_text_node(self.root_)
        sentence_nodes = filter(lambda n: n.nodeType == n.ELEMENT_NODE and n.tagName == 's',
                                list(self.text_node.childNodes))
        sentences = []
        for s in sentence_nodes:
            current = []
            TimeMLDoc.__get_text(s, current, False)
            #print(current)
            if not tag:
                sentences.append(''.join([ c[0] for c in current]))
            else:
                sentences.append(tag_method(current))
        return sentences
                
    def __get_text(element, text, is_event):
        if element.nodeType == element.TEXT_NODE:
            text.append((element.data.replace('\n', ' ').lower(), is_event))
        else:
            for child in element.childNodes:
                if is_event: # the parent is an event
                    TimeMLDoc.__get_text(child, text, is_event)
                elif child.nodeType == child.ELEMENT_NODE:
                    TimeMLDoc.__get_text(child, text, child.tagName == 'EVENT')
                else:
                    # if is just a text node
                    TimeMLDoc.__get_text(child, text, False)
        
