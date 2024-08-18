import cv2
import numpy as np
import math

def distance(p1, p2):
    """Calculer la distance euclidienne entre deux points"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2), abs(p1[0] - p2[0])

def find_connected_components(centroids, max_distance_x, max_distance):
    """Trouver les groupes de points connectés"""
    from sklearn.neighbors import NearestNeighbors
    
    points = np.array(centroids)
    nbrs = NearestNeighbors(n_neighbors=len(points), radius=max_distance).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    visited = [False] * len(points)
    components = []
    
    def dfs(node, component):
        stack = [node]
        while stack:
            current = stack.pop()
            if not visited[current]:
                visited[current] = True
                component.append(points[current])
                neighbors = indices[current]
                for neighbor in neighbors:
                    if not visited[neighbor]:
                        dist, dx = distance(points[current], points[neighbor])
                        if dx < max_distance_x and dist < max_distance:
                            stack.append(neighbor)
    
    for i in range(len(points)):
        if not visited[i]:
            component = []
            dfs(i, component)
            components.append(np.array(component))
    
    return components

def calculate_slope(vx, vy):
    """Calculer la pente d'une ligne en fonction des vecteurs directionnels"""
    if vx == 0:
        return float('inf')  # Ligne verticale
    return vy / vx

def draw_fitting_lines(components, img):
    """Dessiner uniquement les lignes de régression parallèles"""
    slopes = []
    lines = []

    for component in components:
        if len(component) > 1:
            # Convertir les points en format requis par cv2.fitLine
            component = np.array(component, dtype=np.float32)
            [vx, vy, x, y] = cv2.fitLine(component, cv2.DIST_L2, 0, 0.01, 0.01)
            
            # Extraire les valeurs scalaires
            vx = vx[0]
            vy = vy[0]
            x = x[0]
            y = y[0]
            
            # Calculer la pente de la ligne
            slope = calculate_slope(vx, vy)
            
            # Calculer les points de début et de fin de la ligne de régression
            rows, cols = img.shape[:2]
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            
            # Ajouter la pente et les coordonnées de la ligne
            slopes.append(slope)
            lines.append(((cols - 1, righty), (0, lefty)))

    # Supprimer les lignes non parallèles
    unique_slopes = list(set(round(s, 5) for s in slopes))
    
    # Tolérance pour comparer les pentes
    tolerance = 1e-5  # Réduire la tolérance

    parallel_lines = []

    for line, slope in zip(lines, slopes):
        if any(abs(slope - u_slope) < tolerance for u_slope in unique_slopes):
            parallel_lines.append(line)
    
    for line in parallel_lines:
        cv2.line(img, line[0], line[1], (0, 0, 255), 50)  # Dessiner les lignes parallèles 

def row_line(image, result_img):
    # Appliquer un flou pour réduire le bruit
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(image, kernel, iterations=2)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

    # Convertir l'image en niveaux de gris
    grey = cv2.cvtColor(closed, cv2.COLOR_BGR2GRAY)

    # Binariser l'image
    _, binary = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)

    # Trouver les composantes connectées
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Créer une image couleur pour dessiner les lignes entre les centres
    color_image = np.zeros_like(closed)  # Créer une image noire de la même taille que 'closed'

    # Définir les seuils pour la taille des contours et la distance maximale pour relier les centres
    min_contour_area = 200  # Seuil pour la taille minimale du contour (en pixels)
    max_distance_x = 30     # Distance maximale en pixels sur l'axe x pour relier les centres
    max_distance = 2000     # Distance maximale en pixels pour relier les centres

    points = []  # Liste pour stocker les centres des composants

    # Dessiner les lignes entre les centres des composants
    for i in range(1, num_labels):  # Ignorer le label 0 (fond)
        # Vérifier la taille du composant
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_contour_area:
            continue  # Ignorer les petits composants

        # Dessiner le centre du contour
        cx, cy = int(centroids[i][0]), int(centroids[i][1])
        cv2.circle(result_img, (cx, cy), 5, (0, 255, 0), -1)  # Dessiner un cercle vert au centre

        # Ajouter le point au tableau pour le calcul du convex hull
        points.append((cx, cy))

        # Relier les centres des composants proches
        for j in range(i + 1, num_labels):
            if stats[j, cv2.CC_STAT_AREA] < min_contour_area:
                continue  # Ignorer les petits composants
            dist, dx = distance(centroids[i], centroids[j])
            if dx < max_distance_x and dist < max_distance:
                cx2, cy2 = int(centroids[j][0]), int(centroids[j][1])
                #cv2.line(result_img, (cx, cy), (cx2, cy2), (255, 0, 0), 1)  # Dessiner une ligne bleue entre les centres

    # Trouver les groupes de points connectés
    if len(points) > 2:
        components = find_connected_components(points, max_distance_x, max_distance)
        
        # Dessiner les convex hulls et les lignes de régression pour chaque groupe de points connectés
        for component in components:
            if len(component) > 2:
                # Dessiner l'enveloppe convexe
                hull = cv2.convexHull(component)
                area = cv2.contourArea(hull)
                # Seuil d'aire pour filtrer les petites enveloppes convexes
                min_area_threshold = 100000  # Ajustez ce seuil selon vos besoins
                if area > min_area_threshold:
                    cv2.polylines(result_img, [hull], isClosed=True, color=(0, 255, 255), thickness=20)  # Dessiner l'enveloppe convexe en jaune
                
                    # Dessiner les lignes de régression parallèles
                    draw_fitting_lines([component], result_img)

    return result_img

"""# Charger l'image
image = cv2.imread('img/row.png')
result_img = image.copy()
result_img = row_line(image, result_img)

# Sauvegarder l'image résultante
cv2.imwrite('result.png', result_img)"""
