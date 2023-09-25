class Document:
    _id = 0
    def __init__(self, string, emotion):
        self.string = string
        self.emotion = emotion
        self.id = Document._id
        Document.id += 1