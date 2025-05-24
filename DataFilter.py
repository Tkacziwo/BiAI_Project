import ColorAnnotation

class DataFilter:
    def __init__(self):
        self.colorAnnotations = ColorAnnotation.ColorAnnotations()


    def loadColorAnnotations(self, filePath: str):
        colorsArray = ColorAnnotation.ColorsArray()
        i=0

        with open(filePath, 'r') as f:

            for line in f:
                line = line.strip()
                if not line:
                    continue  # pass empty lines

                try:
                    filename, colors_str = line.split(maxsplit=1)
                except ValueError:
                    continue  # skip lines that don't have the expected format

                colors = []

                # Assuming colors are comma-separated
                # and may have leading/trailing spaces
                for c in colors_str.split(','):
                    c = c.strip()
                    if c:
                        # Add color to the ColorsArray
                        colors.append(c)
                
                match i:
                    case 0:
                        colorsArray.setOneColor(colors[0])
                    case 1:
                        colorsArray.setTwoColors(colors[0], colors[1])
                    case 2:
                        colorsArray.setThreeColors(colors[0], colors[1], colors[2])
                    case 3:
                        colorsArray.setFourColors(colors[0], colors[1], colors[2], colors[3])
                    case 4:
                        colorsArray.setFiveColors(colors[0], colors[1], colors[2], colors[3], colors[4])
                
                if i < 4:
                    i += 1
                else:
                    i = 0
                    # Add the colors to the dictionary
                    self.colorAnnotations.addAnnotation(colorsArray, filename)
                    # Reset colorsArray for the next image
                    colorsArray = ColorAnnotation.ColorsArray()

    def filterData(self, variation: float):
        """
        Filter the data based on variation.
        """
        pass

    def getData(self, photo: str):
        """
        Get the filtered data.
        """
        return self.colorAnnotations.getAnnotation(photo)