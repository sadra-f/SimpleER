class Document:
    _id = 0
    def __init__(self, string, emotion):
        self.string = string
        self.emotion = emotion
        self.id = Document._id
        Document._id += 1
    

    def __str__(self) -> str:
        return f"{self.string[0:len(self.string)%11]} - ({self.emotion})"