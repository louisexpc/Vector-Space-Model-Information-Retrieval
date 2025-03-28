import xml.etree.ElementTree as ET
import os
class Query:
    def __init__(self, path):
        """
        Parameters:
        -path: query files path
        """
        if not os.path.exists(path):
            raise ValueError(f"target file {path} doesn't exist")

        try:
            tree = ET.parse(path)
        except Exception as e:
            raise ValueError(f"Load file error: {e}")
        
        self.root = tree.getroot()
        self.topics = self._extraction_topics() #list of dict

    def _extraction_topics(self)->list:
        topics = []
        for topic in self.root.findall("topic"):
            t ={}
            t['number'] =topic.find("number").text.strip() if topic.find("number") is not None else ""
            t['title'] = topic.find("title").text.strip() if topic.find("title") is not None else ""
            t['question'] = topic.find("question").text.strip() if topic.find("question") is not None else ""
            t['narrative'] = topic.find("narrative").text.strip() if topic.find("narrative") is not None else ""
            t['concepts'] = topic.find("concepts").text.strip("。 \n").split("、") if topic.find("concepts") is not None else ""
            topics.append(t)
        return topics
    
    def __repr__(self):
        return f"Query(topics={len(self.topics)})"

if __name__ == "__main__":
    path = os.path.join("queries","query-train.xml")
    q = Query(path)
    print(q)
    print(q.topics)