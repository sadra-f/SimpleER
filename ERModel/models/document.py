class Document:
    """
        Document class meant to hold the text and its respective emotion

        variables
            _id: id counter used to number the inputs
            string: the string which is to be stored
            emotion: the emotion/class representing the string/text
            id: the id of the Document starts from 0 incremented by 1 on each object creation
        
        inputs
            string: the string which is to be stored
            emotion: the emotion/class representing the string/text
    """
    _id = 0
    def __init__(self, string, emotion):
        self.string = string
        self.emotion = emotion
        self.id = Document._id
        Document._id += 1
    

    def __str__(self) -> str:
        return f"{self.string[0:len(self.string)%11]} - ({self.emotion})"