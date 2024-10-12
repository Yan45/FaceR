import face_recognition
from PIL import Image, ImageDraw
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np

#masquer la fenetre
Tk().withdraw()


#Chargement des images de visages à stocker
visage_Emmanuel_Macron = face_recognition.load_image_file("images/E-Macron.jpg")
visage_Michel_Barnier = face_recognition.load_image_file("images/M-Barnier.jpg")
visage_Leon_Marchand = face_recognition.load_image_file("images/L-Marchand.png")

#encodage des visages
encodage_EMacron = face_recognition.face_encodings(visage_Emmanuel_Macron)[0]
encodage_MBarnier = face_recognition.face_encodings(visage_Michel_Barnier)[0]
encodage_LMarchand = face_recognition.face_encodings(visage_Leon_Marchand)[0]

#liste des visages encodes
encodage_visages_stockes = [ encodage_EMacron, encodage_MBarnier, encodage_LMarchand]

#liste des noms des visages stockes
nom_visages_stockes = [ "Emmanuel Macron", "Michel Barnier", "Leon Marchand"]

#ENTREE DE L'IMAGE & ANALYSE

image_location = askopenfilename(title="choisir une image", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])

if image_location:

    image_entree = face_recognition.load_image_file(image_location)   

    #identifier les visages dans l'image + encodage
    location_visage_entre = face_recognition.face_locations(image_entree)
    encodage_visage_entre = face_recognition.face_encodings(image_entree, location_visage_entre)

    #nous allons rendre l'image entree modifiable
    image_pillow = Image.fromarray(image_entree)
    draw = ImageDraw.Draw(image_pillow)

    #comparaison entre les visages encodes stockes et les visages encodes de l'image entree
    for (haut, droite, bas, gauche), encodage_visage_actuel in zip(location_visage_entre, encodage_visage_entre):
        matching = face_recognition.compare_faces(encodage_visages_stockes, encodage_visage_actuel)

        nom = "Inconnu"
        #on compare dans les visages stockes celui qui se rapproche le plus de l'image entrée
        correspondance_visage = face_recognition.face_distance(encodage_visages_stockes, encodage_visage_actuel)
        #on recupere l'indice du visage qui correspond le mieux
        indice_visage = np.argmin(correspondance_visage)

        if matching[indice_visage]: 
            nom = nom_visages_stockes[indice_visage]
    
        draw.rectangle(((gauche, haut), (droite, bas)), outline=(255,0,0))

        #ecrire le n
        # om de la personne correspodante
        draw.text((gauche, bas), nom, (255,255,255))

    image_pillow.show()

    image_pillow.save("Visages_reconnus.jpg")

else: 
    print("Aucune image sélectionnee")








