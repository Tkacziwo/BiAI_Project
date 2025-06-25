
class ColorsArray:

    """
    A class to represent a color array.
    Attributes
    ----------
    OneColor : str
        The first color.
    TwoColors : list
        A list of two colors.
    ThreeColors : list
        A list of three colors.
    FourColors : list
        A list of four colors.
    FiveColors : list
        A list of five colors.
    """

    def __init__(self):
        pass

    def setOneColor(self, color: str):
        
        self.OneColor = [color]

    def setTwoColors(self, color1: str, color2: str):
        
        self.TwoColors = [color1, color2]

    def setThreeColors(self, color1: str, color2: str, color3: str):
        self.ThreeColors = [color1, color2, color3]

    def setFourColors(self, color1: str, color2: str, color3: str, color4: str):
        self.FourColors = [color1, color2, color3, color4]

    def setFiveColors(self, color1: str, color2: str, color3: str, color4: str, color5: str):
        self.FiveColors = [color1, color2, color3, color4, color5] 

    def getOneColor(self):
        return self.OneColor
    
    def getTwoColors(self):
        return self.TwoColors
    
    def getThreeColors(self):
        return self.ThreeColors
    
    def getFourColors(self):
        return self.FourColors
    
    def getFiveColors(self):
        return self.FiveColors


class ColorAnnotations:
    """
    A class to represent a color annotation.

    Attributes
    ----------
    dictionary : dict
        A dictionary to store color annotations for images.
    """

    def __init__(self):

        self.dictionary = {}

    def addAnnotation(self, colorsArray: ColorsArray, image: str):
        if self.dictionary.setdefault(image, None) == None:
            self.dictionary[image] = [colorsArray]
        else:
            self.dictionary[image].append(colorsArray)

    def getAnnotation(self, image: str):
        if image in self.dictionary:
            return self.dictionary[image]
        else:
            return None
        
    def getOneColorAnnotation(self, image: str):
        if image in self.dictionary:
            return self.dictionary[image].getOneColor()
        else:
            return None
        
    def getTwoColorsAnnotation(self, image: str):
        if image in self.dictionary:
            return self.dictionary[image].getTwoColors()
        else:
            return None
        
    def getThreeColorsAnnotation(self, image: str):
        if image in self.dictionary:
            return self.dictionary[image].getThreeColors()
        else:
            return None
        
    def getFourColorsAnnotation(self, image: str):
        if image in self.dictionary:
            return self.dictionary[image].getFourColors()
        else:
            return None
        
    def getFiveColorsAnnotation(self, image: str):
        if image in self.dictionary:
            return self.dictionary[image].getFiveColors()
        else:
            return None

    def getAllAnnotations(self):
        return self.dictionary