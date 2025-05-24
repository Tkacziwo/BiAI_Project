import ColorAnnotation
from sklearn.cluster import DBSCAN, KMeans as Kmeans
from collections import defaultdict
import numpy as np

def hex_to_rgb_vector( hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    return [r, g, b]

class DataFilter:
    def __init__(self):
        self.colorAnnotations = ColorAnnotation.ColorAnnotations()
        self.filteredData = ColorAnnotation.ColorAnnotations()

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

    def filterData(self,):
        """
        Filter the data based on variation.
        """
        """for img in self.colorAnnotations.dictionary:
            colors = self.colorAnnotations.dictionary[img]
            if len(colors) > 1:
                # Apply DBSCAN clustering to filter colors
                dbscan = DBSCAN(eps=variation, min_samples=1)
                color_vectors = [hex_to_rgb_vector(c.getOneColor()[0]) for c in colors]
                labels = dbscan.fit_predict(color_vectors)
                
                # Group colors by cluster
                clustered_colors = {}
                for label, color in zip(labels, colors):
                    if label not in clustered_colors:
                        clustered_colors[label] = []
                    clustered_colors[label].append(color)

                # Keep only the most representative color from each cluster
                filtered_colors = [cluster[0] for cluster in clustered_colors.values()]
                self.colorAnnotations.dictionary[img] = filtered_colors"""
        clustered_results = {}  # końcowy słownik: {obraz: {liczba_kolorów: list_kolorów}}
        min_samples = 1  # minimalna liczba próbek w klastrze

        for image, annotations in self.colorAnnotations.dictionary.items():
            color_groups = defaultdict(list)

            # Zbieramy kolory wg liczby w zbiorze
            for annotation in annotations:
                if hasattr(annotation, "getOneColor"):
                    color_groups[1].extend(annotation.getOneColor())
                if hasattr(annotation, "getTwoColors"):
                    color_groups[2].extend(annotation.getTwoColors())
                if hasattr(annotation, "getThreeColors"):
                    color_groups[3].extend(annotation.getThreeColors())
                if hasattr(annotation, "getFourColors"):
                    color_groups[4].extend(annotation.getFourColors())
                if hasattr(annotation, "getFiveColors"):
                    color_groups[5].extend(annotation.getFiveColors())

            clustered_results[image] = {}

            # Dla każdej grupy (1,2,...,5 kolorów)
            for k, hex_colors in color_groups.items():
                if len(hex_colors) < min_samples:
                    continue  # za mało punktów, by klasteryzować

                #klasteryzacja kmeans
                rgb_colors = np.array([hex_to_rgb_vector(c) for c in hex_colors])
                clustering = Kmeans(n_clusters=k, random_state=42).fit(rgb_colors)
                centroids = clustering.cluster_centers_
                clustered_results[image][k] = centroids
                """step = 10000**(1/k)  # krok dla eps w DBSCAN

                #klasteryzacja dbscan
                for i in range(1, 100000, int(step)):
                    rgb_colors = np.array([hex_to_rgb_vector(c) for c in hex_colors])
                    clustering = DBSCAN(eps=i/10000, min_samples=min_samples).fit(rgb_colors)

                    labels = clustering.labels_
                    unique_labels = set(labels) - {-1}  # odrzucamy outliery
                    if len(unique_labels) == k:
                        break"""
                """rgb_colors = np.array([hex_to_rgb_vector(c) for c in hex_colors])
                clustering = DBSCAN(eps=1**(-100**k), min_samples=min_samples).fit(rgb_colors)

                labels = clustering.labels_
                unique_labels = set(labels) - {-1}  # odrzucamy outliery
                

                # Wylicz centroidy (średnie RGB)
                clustered_rgb = []
                for label in unique_labels:
                    points = rgb_colors[labels == label]
                    centroid = np.mean(points, axis=0)
                    clustered_rgb.append(centroid)

                clustered_results[image][k] = clustered_rgb """
            self.filteredData.dictionary[image] = clustered_results[image]
        

    def getData(self, photo: str):
        """
        Get the filtered data.
        """
        return self.filteredData.getAnnotation(photo)
    
    def getOneColorAnnotation(self, photo: str):
        return self.colorAnnotations.getOneColorAnnotation(photo)

    def getDictionary(self):
        return self.colorAnnotations