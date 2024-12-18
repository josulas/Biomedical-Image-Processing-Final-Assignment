import streamlit as st
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import pandas as pd
import cv2
from segmentation.k_means import kmeans, KmeansFlags, KmeansTermCrit, KmeansTermOpt
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_elements_from_indexes, TRAIN_DATASET, LABELS
from improving.filtering import conv2d
from streamlit_drawable_canvas import st_canvas
from PIL import Image

image_file_path= None
# base de datos de doctores
USER_DATA = {
    "1": "1",
    "doctor2@example.com": "securepass456"
}

# Estado de inicio de sesión
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# Función para mostrar la página de login
def login_page():
    st.image('mri.jpg', use_column_width=True)
    st.title("Bienvenidx Doctor/a!!") 

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    login_clicked = st.button("Sign In")

    if login_clicked:
        if email in USER_DATA and USER_DATA[email] == password:
            st.session_state.logged_in = True
            st.success("Inicio de sesión exitoso!")
            #menu_page()
        else:
            st.error("Email o contraseña incorrectos")
            #login_page()


def subir_imagen(filename):
    global image_file_path
    upload_folder = r"interface/images_database"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    uploaded_file = st.file_uploader("Elija la imagen del paciente correspondiente:", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Extensión original del archivo
        extension = os.path.splitext(uploaded_file.name)[1]

        if filename:
            # Leer la imagen
            bytes_data = uploaded_file.getvalue()
            image_file_path = os.path.join(upload_folder, filename + extension) 
            
            with open(image_file_path, "wb") as f:
                f.write(bytes_data)

            st.write(f"🙌🏼 ✅ Su imagen ha sido subida con éxito")
            st.write(image_file_path)
        else:
            st.warning("Por favor, ingrese un nombre para la imagen.")


def alta_paciente(name, lastname, DNI, email,doctor):
    # Conectar a la base de datos (se creará automáticamente si no existe)
    conn = sqlite3.connect(r'interface/pacientes.db')
    c = conn.cursor()

    # Crear la tabla si no existe
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT NOT NULL,
            last_name TEXT NOT NULL,
            dni INT,
            email TEXT,
            doctor TEXT
        )
    ''')
    conn.commit()

    # Insertar solo si todos los campos están completos
    if name and lastname and DNI and email and doctor:
        c.execute('INSERT INTO users (first_name, last_name, dni, email, doctor) VALUES (?, ?, ?, ?, ?)', 
                  (name, lastname, DNI, email, doctor))
        conn.commit()

 

def mostrar_paciente():
    conn = sqlite3.connect(r'interface/pacientes.db')
    c = conn.cursor()

    c.execute("SELECT first_name, last_name, dni, email FROM users")
    rows = c.fetchall()

    if rows:
        df = pd.DataFrame(
            rows,
            columns=["Nombre", "Apellido", "DNI", "Email"]
        )

        st.dataframe(df, hide_index=True)

    else:
        st.write("No hay usuarios registrados aún.")

def draw_grayscale_image(image, ax):
    ax.imshow(image, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

def filter(imagen):

    epsilon = 0.1 
    sigma = 3
    gaussianDim = int(np.ceil(np.sqrt(-2 * sigma ** 2 * np.log(epsilon * sigma * np.sqrt(2 * np.pi)))))
    gaussianKernel1D = cv2.getGaussianKernel(gaussianDim, sigma)
    gaussianKernel = np.outer(gaussianKernel1D, gaussianKernel1D)
    filtered_image = conv2d(imagen, gaussianKernel)

    return filtered_image


def menu_results(option):
    st.write(image_file_path)
    if option == "Segmentación Automática":
        k_means= st.toggle("Segmentar con K-means")
        ratio= st.toggle("Calcular proporción materia gris sobre materia blanca")

        st.markdown(
    """
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #d6e9f7;">
        <h4>💡 Por qué puede ser interesante?</h4>
        <p style="margin: 0;">En condiciones normales, la proporción entre sustancia gris (40%) sobre sustancia blanca (60%) en el cerebro es aproximadamente del 66,6%. En pacientes con Alzheimer, esta proporción disminuye progresivamente debido a la atrofia de la sustancia gris y la degeneración de la sustancia blanca. La reducción de esta relación se asocia con un mayor grado de demencia y deterioro cognitivo, ya que afecta funciones clave como la memoria, el razonamiento y las habilidades para realizar tareas diarias. Este marcador puede utilizarse como indicador del avance de la enfermedad mediante estudios de neuroimagen.</p>
    </div>
    """,
    unsafe_allow_html=True)
        st.write("  ")
    
        if st.button("Segmentar!"):
            img = cv2.imread(image_file_path, cv2.IMREAD_GRAYSCALE)

            img = filter(img) #filtramos
            compactness, labels, centers= kmeans(img.flatten(), 3,attempts=5)
            centers = centers.astype(np.uint8)
            segmented_kmeans = centers[labels].reshape(img.shape)
            sorted_centers = sorted(centers)
            background_idx = np.argmax(centers == sorted_centers[0])
            grey_matter_idx = np.argmax(centers == sorted_centers[1])
            white_matter_idx = np.argmax(centers == sorted_centers[2])
            segmented_white_matter = np.where(segmented_kmeans == centers[white_matter_idx], 1, 0)
            segmented_grey_matter = np.where(segmented_kmeans == centers[grey_matter_idx], 1, 0)
            segmented_background = np.where(segmented_kmeans == centers[background_idx], 1, 0)
            proportion= np.round(np.sum(segmented_grey_matter)/np.sum(segmented_white_matter) , 3)
            porcentaje_gris= np.round(np.sum(segmented_grey_matter)*100/(np.sum(segmented_grey_matter)+np.sum(segmented_white_matter)) , 2)
            

            if ratio:
                st.text("El resultado de la proporción materia gris/ materia blanca es:")
                st.code(proportion, 'plaintext')
                st.text("El porcentaje de materia gris (%) es:")
                st.code(porcentaje_gris, 'plaintext')

            if k_means:
                if 'fig_kmeans' in globals():
                    plt.close('Kmeans results')

                fig_kmeans, axs_kmeans = plt.subplots(2, 2, num='Kmeans results')
                draw_grayscale_image(segmented_kmeans, axs_kmeans[0][0])
                draw_grayscale_image(segmented_white_matter * 255, axs_kmeans[0][1])
                draw_grayscale_image(segmented_grey_matter * 255, axs_kmeans[1][0])
                draw_grayscale_image(segmented_background * 255, axs_kmeans[1][1])
                fig_kmeans.tight_layout()
                st.pyplot(fig_kmeans)

            path, ext= os.path.splitext(image_file_path)
            new_path=f"{path}_seg{ext}"
            img_segmented= cv2.imwrite(new_path, segmented_kmeans)

    
    elif option == "Segmentación Manual (draw)":
        if image_file_path is not None:
            image = Image.open(image_file_path)
            st.subheader("Dibuje y pinte la region que desee segmentar:", anchor= "seg_man", divider= 'grey')
            
            lapiz = st.slider("Selecciona el grosor del trazo:", min_value=3, max_value=50, value=20)
            canvas_result = st_canvas(fill_color="rgba(255, 0, 0, 0.3)", stroke_width=lapiz, stroke_color="red", background_image=image,height=image.height, width=image.width, drawing_mode="freedraw",key="canvas")

            if canvas_result.image_data is not None:
                st.image(canvas_result.image_data, caption="Segmentación manual", use_column_width=True)
                canvas_image = Image.fromarray((canvas_result.image_data).astype("uint8"))
            
            #guardamos el dibujo
                path, ext= os.path.splitext(image_file_path)
                new_path=f"{path}_mask{ext}"
                canvas_image.save(new_path)

                mascara = np.array(canvas_result.image_data[:, :, 3]) 
                mascara_bin = np.where(mascara > 0, 255, 0).astype("uint8") 
            
            
                original_array = np.array(image)
                segmentada = np.copy(original_array)
                segmentada[mascara_bin == 0] = 0 
                st.image(segmentada, caption="Imagen segmentada", use_column_width=True)
                segmentada= cv2.imwrite(f"{path}_manualseg{ext}", segmentada)

    elif option == "Clasificador Avanzado":
       st.write("hola") 

# INICIOOOOOO
if 'paciente_guardado' not in st.session_state:
    st.session_state.paciente_guardado = False
    st.session_state.archivo = None

def menu_page():
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Inicio", "👥 Agregar Paciente", "🔎 Buscar Paciente", " 📊 Ver Resultados"])

    with tab1:
        tab1.header(":blue[Bienvenido Doctor!!]👩🏻‍⚕️👨🏻‍⚕️")
        tab1.markdown("Elija la :blue[**opción correspondiente**] para comenzar con las actividades:")
        tab1.markdown("""El primer paso es cargar los datos del paciente en :orange[**"agregar paciente"**] para darlo de alta y después *subir la imagen*.<br><br>
                      Luego en la pestaña de :orange[**"buscar pacientes"**] se pueden hallar los que ya fueron ingresados.<br><br>
                      Para analizar la imágen se debe dirigir a la pestaña de :orange[**"ver resultados"**] donde hallará las diferentes opciones para analizar la imagen ingresada
                      """, unsafe_allow_html=True)
        

    with tab2:
        tab2.subheader("Alta de pacientes")
        tab2.markdown("Ingrese los datos del nuevo paciente para cargar la imagen:")

        name = st.text_input("Nombre")
        lastname = st.text_input("Apellido")
        DNI = st.text_input("DNI")
        email = st.text_input("Dirección de correo")
        doctor = st.text_input("Doctor")

        # Botón para guardar datos del paciente
        if st.button("Guardar Paciente"):
            if name and lastname and DNI and email and doctor: #todos los datos completos
                # Guardamos datos del paciente
                alta_paciente(name, lastname, DNI, email, doctor)
                archivo = f"{lastname}_{DNI}"
                st.session_state.archivo = archivo
                st.session_state.paciente_guardado = True  # Marcamos que el paciente fue guardado
                st.success("Paciente agregado exitosamente. Ahora puedes subir la imagen.")
            else:
                st.warning("Por favor, complete todos los datos antes de guardar.")

        # Solo mostramos el cargador de imágenes si el paciente ha sido guardado
        if st.session_state.paciente_guardado:
            subir_imagen(st.session_state.archivo)

    with tab3:
        mostrar_paciente()
    with tab4:
        option = tab4.radio("Selecciona una opción para proceder con el análisis de la imágen", ("Segmentación Manual (draw)", "Segmentación Automática","Clasificador Avanzado"))
        menu_results(option)


if st.session_state.logged_in:
    menu_page()
else:
    login_page()
    