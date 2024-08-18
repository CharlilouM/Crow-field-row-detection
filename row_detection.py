import cv2
import numpy as np
import filtre

# Charger l'image
image = cv2.imread('img/2.jpg')

# Convertir l'image en espace de couleurs HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define the list of boundaries in HSV
boundaries = [
    # Green range (HSV)
    ([40, 40, 40], [70, 255, 255]),  # Ajuster les valeurs selon vos besoins
    # Brown range (HSV)
    ([10, 100, 20], [20, 255, 200])  # Ajuster les valeurs selon vos besoins
]

# Liste pour stocker les images filtrées
filtered_images = []

# Loop over the boundaries
for (lower, upper) in boundaries:
    # Créer des tableaux NumPy pour les limites inférieures et supérieures
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # Trouver les couleurs dans les limites spécifiées et appliquer le masque
    mask = cv2.inRange(hsv_image, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # Ajouter l'image filtrée à la liste
    filtered_images.append(output)

# Empiler les images côte à côte
image_filtre=filtre.row_line(filtered_images[1],image.copy())

# Afficher les images
cv2.imshow("Images Originale et Filtrées", np.hstack([image] + [image_filtre]))
cv2.waitKey(0)
cv2.destroyAllWindows()
