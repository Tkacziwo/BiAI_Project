import ColorAnnotation
from sklearn.cluster import DBSCAN, KMeans as Kmeans
from collections import defaultdict
import numpy as np
from collections import Counter
from skimage import color

def hex_to_lab_vector( hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
    rgb_color = np.array([[[r, g, b]]], dtype=np.float32)  # trzeba mieć 3D shape

    rgb_normalized = rgb_color / 255.0

    lab_color = color.rgb2lab(rgb_normalized)
    l, a, b = lab_color[0][0]  # wyciągamy wartości L, a, b z 3D array
    return [l, a, b]

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def show_lab_clusters(img, colors_count, lab_points, labels=None, centroids=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    lab_points = np.array(lab_points)

    if labels is None:
        labels = np.zeros(len(lab_points), dtype=int)
    
    labels = np.array(labels)
    unique_labels = np.unique(labels)

    labels = np.array(labels)
    lab_points = np.array(lab_points)
    #assert len(labels) == lab_points.shape[0]

    unique_labels = np.unique(labels)

    for label in unique_labels:
        mask = labels == label
        cluster_points = lab_points[mask]

        if len(cluster_points) == 0:
            continue  # opcjonalnie: pomiń puste

        ax.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            cluster_points[:, 2],
            label=f'Cluster {label}',
            s=50,
            alpha=0.7,
            edgecolors='k'
        )

    # Dodaj centroidy jeśli są
    if centroids is not None:
        centroids = np.array(centroids)
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            centroids[:, 2],
            c='black',
            marker='X',
            s=200,
            edgecolors='white',
            label='Centroids'
        )

    ax.set_xlabel('L')
    ax.set_ylabel('A')
    ax.set_zlabel('B')
    ax.set_title(f'{img} clusters: {colors_count} colors')
    ax.legend()
    plt.tight_layout()
    plt.show()



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
        min_samples = [3, 3, 2, 2, 2]#int(len(hex_colors) / 2 / k)  # minimalna liczba próbek w klastrze
        eps = [0.04, 0.035, 0.025, 0.02, 0.018]  # odległość dla DBSCAN, można dostosować
        epsForSpecialImage = [[0.02, 0.01, 0.01, 0.01, 0.01],       # specjalny przypadek dla obrazu "00000050735.jpg"
                              [0.02, 0.02, 0.03, 0.025, 0.02],      # specjalny przypadek dla obrazu "000000083368.jpg" za dużo klastrów
                              [0.04, 0.02, 0.018, 0.02, 0.018],     # specjalny przypadek dla obrazu "000000120148.jpg" zbyt rozpięty 3 kolory
                              [0.02, 0.02, 0.01, 0.01, 0.01],       # specjalny przypadek dla obrazu "000000149833.jpg" 1 i 2 i pozostałe zbyt rozległe klastry
                              [0.02, 0.02, 0.03, 0.03, 0.025],      # specjalny przypadek dla obrazu "000000203107.jpg" za dużo klastrów dla 4 i 5 kolorów
                              [0.045, 0.02, 0.02, 0.01, 0.01],      # specjalny przypadek dla obrazu "000000299394.jpg" za mało klastrów dla 2 kolorów
                              [0.02, 0.02, 0.02, 0.02, 0.02],       # specjalny przypadek dla obrazu "000000302008.jpg" rozległy klaster dla 1 koloru
                              [0.04, 0.035, 0.025, 0.02, 0.03],     # specjalny przypadek dla obrazu "000000331712.jpg" za dużo klastrów dla 5 kolorów
                              [0.03, 0.035, 0.03, 0.03, 0.03],      # specjalny przypadek dla obrazu "000000354425.jpg" za dużo klastrów od 3 w zwyź
                              [0.04, 0.035, 0.025, 0.02, 0.03],     # specjalny przypadek dla obrazu "000000367461.jpg" dużo klastrów dla 5 kolorów
                              [0.04, 0.035, 0.025, 0.02, 0.03],     # specjalny przypadek dla obrazu "000000382552" za dużo dla 5 kolorów
                              [0.03, 0.02, 0.02, 0.02, 0.018],      # specjalny przypadek dla obrazu "000000389092.jpg" za szerokie klastry (2,3 kol)
                              [0.04, 0.035, 0.025, 0.02, 0.02],     # specjalny przypadek dla obrazu "000000389699.jpg" dużo klastrów 5 kol
                              [0.02, 0.02, 0.01, 0.01, 0.01],       # specjalny przypadek dla obrazu "000000393443.jpg" 1, 2, 3, 4, 5 kolory rozległy klaster
                              [0.04, 0.035, 0.025, 0.03, 0.03],     # specjalny przypadek dla obrazu "000000403265.jpg" 4, 5 kolorów za dużo klastrów
                              [0.04, 0.035, 0.025, 0.02, 0.03],     # specjalny przypadek dla obrazu "000000447099.jpg" 5 kolorów za dużo klastrów
                              [0.04, 0.035, 0.025, 0.03, 0.022],    # specjalny przypadek dla obrazu "000000452186" 4 kolory za dużo klastrów
                              [0.04, 0.035, 0.035, 0.03, 0.025],    # specjalny przypadek dla obrazu "000000452354.jpg" 3,4,5 kolorów za dużo klastrów
                              [0.04, 0.025, 0.02, 0.02, 0.018],     # specjalny przypadek dla obrazu "000000461404.jpg" 2 kolory rozległy klaster, 3 za mało
                              [0.04, 0.035, 0.02, 0.02, 0.015],     # specjalny przypadek dla obrazu "000000471452.jpg" za dużo klastrów 3,4, 5
                              [0.04, 0.025, 0.025, 0.03, 0.025],    # specjalny przypadek dla obrazu "000000475330.jpg" 2 kolory za mało, 4 za dużo, 5 za dużo
                              [0.04, 0.035, 0.025, 0.02, 0.025],    # specjalny przypadek dla obrazu "000000489541.jpg" 5 kolorów za dużo
                              [0.01, 0.01, 0.01, 0.01, 0.01],       # specjalny przypadek dla obrazu "000000535871.jpg" 1 kolor za rozległy, 2, 3, 4, 5 też
                              [0.03, 0.035, 0.025, 0.02, 0.025],    # specjalny przypadek dla obrazu "000000537131.jpg" 1 klaster rozległy, 5 za dużó
                              [0.04, 0.035, 0.025, 0.03, 0.025],    # specjalny przypadek dla obrazu "000000547938.jpg" 4, 5 kolory za dużo
                              [0.04, 0.035, 0.025, 0.02, 0.015]]    # 000000556123.jpg - 5 kolorów zbyt rozległy klaster 

        clustered_results = {}  # końcowy słownik: {obraz: {liczba_kolorów: list_kolorów}}
        specialImagesCount = 0
        specialImages= ["000000050735.jpg", "000000083368.jpg", "000000120148.jpg", "000000149833.jpg", "000000203107.jpg", "000000299394.jpg", "000000302008.jpg", "000000331712.jpg", "000000354425.jpg", "000000367461.jpg", "000000382552.jpg", "000000389092.jpg", "000000389699.jpg", "000000393443.jpg", "000000403265.jpg", "000000447099.jpg", "000000452186.jpg", "000000452354.jpg", "000000461404.jpg", "000000471452.jpg", "000000475330.jpg", "000000489541.jpg", "000000535871.jpg", "000000537131.jpg", "000000547938.jpg", "000000556123.jpg"]

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
                
                if len(hex_colors) < min_samples[k-1]:
                    continue  # za mało punktów, by klasteryzować

                # Konwersja kolorów HEX do LAB
                lab_colors = np.array([hex_to_lab_vector(c) for c in hex_colors])

                #klasteryzacja kmeans
                """lab_colors = np.array([hex_to_lab_vector(c) for c in hex_colors])
                clustering = Kmeans(n_clusters=k, random_state=42).fit(lab_colors)
                #show_lab_clusters(lab_colors, clustering.labels_, clustering.cluster_centers_)
                centroids = clustering.cluster_centers_
                clustered_results[image][k] = centroids"""

                # klasteryzacja DBSCAN
                if image in specialImages:
                    clusteringDB = DBSCAN(eps=epsForSpecialImage[specialImagesCount][k-1], min_samples=2).fit(lab_colors)
                    labelsDB = clusteringDB.labels_
                    show_lab_clusters(image, k, lab_colors, labelsDB)
                else:
                    clusteringDB = DBSCAN(eps=eps[k-1], min_samples=min_samples[k-1]).fit(lab_colors)

                    labelsDB = clusteringDB.labels_
                #show_lab_clusters(image, k, lab_colors, labelsDB)
                #unique_labels = np.array(labelsDB) - [-1]  # odrzucamy outliery

                mask, selected_labels = self.get_largest_k_clusters(labelsDB, k)

                # Wylicz centroidy (średnie LAB)
                clustered_lab = []
                for label in selected_labels:
                    points = lab_colors[labelsDB == label]
                    centroid = np.mean(points, axis=0)
                    clustered_lab.append(centroid)

                clustered_results[image][k] = clustered_lab 
            if image in specialImages:
                specialImagesCount += 1
            self.filteredData.dictionary[image] = clustered_results[image]
        

    def getData(self, photo: str):
        """
        Get the filtered data.
        """
        return self.filteredData.getAnnotation(photo)
    
    

    def get_largest_k_clusters(self, labels, k):
        labels = np.array(labels)
        
        # Policz ile punktów ma każdy klaster (z pominięciem -1 jeśli istnieje)
        label_counts = Counter(label for label in labels if label != -1)

        # Wybierz k największych klastrów
        largest_labels = [label for label, _ in label_counts.most_common(k)]

        # Zwróć maskę logiczną dla punktów należących do tych klastrów
        mask = np.isin(labels, largest_labels)

        return mask, largest_labels