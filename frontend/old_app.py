import streamlit as st
import sys
import os
import time
from streamlit.delta_generator import DeltaGenerator
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import sqlite3
import pandas as pd
import cv2
from segmentation.k_means import kmeans
import numpy as np
import matplotlib.pyplot as plt
from model.dataset import LABELS
from improving.filtering import conv2d
from streamlit_drawable_canvas import st_canvas
from PIL import Image
from model.inference import build_and_load_model_from_state, pre_process_image, predict, PREDICTION_POWER
import re

# Global variables definitions
if "confirm" not in st.session_state: # Used to visualize results
    st.session_state.confirm = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'patient_saved' not in st.session_state:
    st.session_state.patient_saved = False
if "selected_dni" not in st.session_state:
    st.session_state.selected_dni = None
if "selection_state" not in st.session_state:
    st.session_state.selection_state = {}
if "patient_data" not in st.session_state:
    st.session_state.patient_data = {"DNI": None, "Nombre": None, "Apellido": None, "Email": None,
                                     "image_file_path": None}
if "new_filename" not in st.session_state:
    st.session_state.new_filename = None
if "delete_data" not in st.session_state:
    st.session_state.delete_data = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "reload_image" not in st.session_state:
    st.session_state.reload_image = None


upload_folder = r"./interface/images_results"

# Doctors' database`
USER_DATA = {
    "sbouzo@itba.edu.ar": "sofi_es_doctora",
    "1": "1",
    # "doctor2@example.com": "securepass456"
}


def get_patients_filename(patients_surname, patients_id):
    return f"{patients_surname}_{patients_id}"


def validate_id(patients_id: str) -> bool:
    if patients_id.isdigit() and 1_000_000 <= int(patients_id) <= 99_999_999:
        return True
    else:
        return False


def validate_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def find_file_with_extension(directory, filename):
    # List all files in the directory
    for file in os.listdir(directory):
        # Check if the file starts with the given filename (ignoring extension)
        if os.path.splitext(file)[0] == filename:
            return file  # Return the full filename with the extension
    return None  # Return None if no match is found


def login_page() -> None:
    """
    Function to display login screen
    @return:
    None
    """
    st.image(r'./interface/login_image.jpg', use_column_width=True)
    st.title("Inicio de Sesión")
    email = st.text_input("Email")
    password = st.text_input("Contraseña", type="password")
    if st.button("Iniciar Sesión"):
        if email in USER_DATA and USER_DATA[email] == password:
            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.success("¡Inicio de sesión exitoso!")
            time.sleep(2)
            st.rerun()
        else:
            st.error("Email o contraseña incorrectos")


def subir_imagen(filename, key = None):
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    uploaded_file = st.file_uploader("Elija la imagen del paciente correspondiente:", type=["png", "jpg", "jpeg"],
                                     key = key)

    if uploaded_file is not None:
        # Extensión original del archivo
        extension = os.path.splitext(uploaded_file.name)[1]
        # st.write(uploaded_file.name)
        if filename:
            # Leer la imagen
            bytes_data = uploaded_file.getvalue()
            st.session_state.patient_data["image_file_path"] = os.path.join(upload_folder, filename + extension)

            with open(st.session_state.patient_data["image_file_path"], "wb") as f:
                f.write(bytes_data)

            st.write(f"🙌🏼 ✅ Su imagen ha sido subida con éxito")
            
        else:
            st.warning("Por favor, ingrese un nombre para la imagen.")


def alta_paciente(name, lastname, DNI, email, doctor):
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
    dnis = [row[0] for row in c.execute('SELECT dni FROM users').fetchall()]
    if int(DNI) in dnis:
        st.error("Este paciente ya fue agregado")
    else:
        if name and lastname and DNI and email and doctor:
            c.execute('INSERT INTO users (first_name, last_name, dni, email, doctor) VALUES (?, ?, ?, ?, ?)',
                      (name, lastname, DNI, email, doctor))
            conn.commit()
        st.success("Paciente agregado exitosamente. Ahora puedes subir la imagen.")
        for key in st.session_state.selection_state:
            st.session_state.selection_state[key] = False
        st.session_state.selection_state[int(DNI)] = True
 

def mostrar_paciente():
    conn = sqlite3.connect(r'interface/pacientes.db')
    c = conn.cursor()

    c.execute("SELECT first_name, last_name, dni, email, doctor FROM users")
    rows = c.fetchall()

    selected_dni = st.session_state.get("selected_dni", None)

    if rows:
        # Convert to a DataFrame
        df = pd.DataFrame(
            rows,
            columns=["Nombre", "Apellido", "DNI", "Email", "Doctor"]
        )
        # df.set_index('DNI', inplace=True)
        df["Selected"] = df["DNI"].apply(lambda dni: st.session_state.selection_state.get(dni, False))
        edited_df = st.data_editor(
            df,
            column_config={
                "Selected": st.column_config.CheckboxColumn(
                    "Selected",
                    default=False,
                )
            },
            # disabled=["widgets"],
            hide_index=True,
        )

        selected_rows = edited_df[edited_df["Selected"]]
        if not selected_rows.empty:
            for _, row in selected_rows.iterrows():
                if selected_dni != row["DNI"]:
                    if selected_dni is not None:
                        edited_df.loc[edited_df[edited_df["DNI"] == selected_dni].index[0], "Selected"] = False
                        st.session_state.selection_state[st.session_state.selected_dni] = False
                    st.session_state.selected_dni = row["DNI"]
                    st.session_state.selection_state[row["DNI"]] = True
                    st.session_state.confirm = None
                    st.session_state.patient_data["DNI"] = row["DNI"]
                    st.session_state.patient_data["Apellido"] = row["Apellido"]
                    st.session_state.patient_data["Email"] = row["Email"]
                    st.session_state.patient_data["Nombre"] = row["Nombre"]
                    image_path = os.path.join(upload_folder,
                                              find_file_with_extension(upload_folder,
                                                                       get_patients_filename(row["Apellido"],
                                                                                             row["DNI"])))
                    if os.path.exists(image_path):
                        st.session_state.patient_data["image_file_path"] = image_path
                    else:
                        st.session_state.patient_data["image_file_path"] = None
                    st.write(st.session_state.patient_data["image_file_path"])
                    st.session_state.reload_image = False
                    st.rerun()
            if st.button("Volver a cargar la imagen del paciente seleccionado"):
                st.session_state.reload_image = True
            if st.session_state.reload_image:
                row = df[df["DNI"] == st.session_state.selected_dni]
                st.session_state.new_filename = get_patients_filename(row["Apellido"][0], row["DNI"][0])
                subir_imagen(st.session_state.new_filename, "reload")
            if st.button("Borrar datos del paciente seleccionado"):
                st.session_state.delete_data = True
            if st.session_state.delete_data:
                st.warning("Esta acción es irreversible.")
                if st.button("Confirmar selección", key="delete_patient"):
                    st.session_state.delete_data = False
                    c.execute("DELETE FROM users WHERE dni = ?", (st.session_state.selected_dni,))
                    conn.commit()
                    st.session_state.patient_data["DNI"] = None
                    st.session_state.patient_data["Apellido"] = None
                    st.session_state.patient_data["Email"] = None
                    st.session_state.patient_data["Nombre"] = None
                    st.session_state.patient_data["image_file_path"] = None
                    del st.session_state.selection_state[st.session_state.selected_dni]
                    st.session_state.confirm = None
                    st.session_state.selected_dni = None
                    st.info("Datos Borrados Exitosamente")
                    time.sleep(1)
                    st.rerun()
        else:
            st.session_state.patient_data["DNI"] = None
            st.session_state.patient_data["Apellido"] = None
            st.session_state.patient_data["Email"] = None
            st.session_state.patient_data["Nombre"] = None
            st.session_state.patient_data["image_file_path"] = None
            st.session_state.reload_image = False
            if st.session_state.selected_dni is not None:
                st.session_state.selection_state[selected_dni] = False
                st.session_state.selected_dni = None
                st.rerun()



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
    if st.session_state.patient_data["image_file_path"] is None:
        return

    if option == "Segmentación Automática":
        st.subheader("Segmentación Automática con K-means", anchor= "seg_autom", divider= 'grey')
        k_means= st.toggle("Segmentar con K-means")
        ratio= st.toggle("Calcular proporción materia gris sobre materia blanca")

        st.markdown(
    """
    <div style="background-color: #f0f8ff; padding: 15px; border-radius: 10px; border: 1px solid #d6e9f7; color: black;">
        <h4 style="color: black">💡 ¿Por qué puede ser interesante?</h4>
        <p style="margin: 0;">En condiciones normales, la proporción entre sustancia gris (40%) sobre sustancia blanca (60%) en el cerebro es aproximadamente del 66,6%. En pacientes con Alzheimer, esta proporción disminuye progresivamente debido a la atrofia de la sustancia gris y la degeneración de la sustancia blanca. La reducción de esta relación se asocia con un mayor grado de demencia y deterioro cognitivo, ya que afecta funciones clave como la memoria, el razonamiento y las habilidades para realizar tareas diarias. Este marcador puede utilizarse como indicador del avance de la enfermedad mediante estudios de neuroimagen.</p>
    </div>
    """,unsafe_allow_html=True)
        st.write("  ")
    
        if st.button("Segmentar"):
            img = cv2.imread(st.session_state.patient_data["image_file_path"], cv2.IMREAD_GRAYSCALE)

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
            area_total = np.round((np.sum(segmented_grey_matter)+np.sum(segmented_white_matter))/(img.shape[1]*img.shape[0]) * 24 ** 2,2)

            if ratio:
                st.text("El resultado de la proporción materia gris/ materia blanca es:")
                st.code(proportion, 'plaintext')
                st.text("El porcentaje de materia gris (%) es:")
                st.code(porcentaje_gris, 'plaintext')
                st.text("El area total de materia gris y blanca es (cm2):")
                st.code(area_total, 'plaintext')

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

            path, ext= os.path.splitext(st.session_state.patient_data["image_file_path"])
            new_path=f"{path}_seg{ext}"
            img_segmented = cv2.imwrite(new_path, segmented_kmeans)

    
    elif option == "Segmentación Manual (draw)":
        image_og = Image.open(st.session_state.patient_data["image_file_path"])
        image = image_og.resize((4*image_og.width, 4*image_og.height), Image.Resampling.LANCZOS)
        st.subheader("Dibuje y pinte la region que desee segmentar:", anchor= "seg_man", divider= 'grey')

        lapiz = st.slider("Selecciona el grosor del trazo:", min_value=3, max_value=50, value=20)
        canvas_result = st_canvas(fill_color="rgba(255, 0, 0, 0.3)", stroke_width=lapiz, stroke_color="red", background_image=image, height=image.height, width=image.width, drawing_mode="freedraw",key="canvas")

        if canvas_result.image_data is not None:
            st.image(canvas_result.image_data, caption="Segmentación manual", use_column_width=True)
            canvas_image = Image.fromarray((canvas_result.image_data).astype("uint8"))

        #guardamos el dibujo
            path, ext= os.path.splitext(st.session_state.patient_data["image_file_path"])
            new_path=f"{path}_mask{ext}"
            canvas_image.save(new_path)

            mascara = np.array(canvas_result.image_data[:, :, 3])
            mascara_bin = np.where(mascara > 0, 255, 0).astype("uint8")


            image_array = np.array(image.convert("RGBA"))
            segmentada = np.copy(image_array)
            area = np.sum(mascara_bin) / 255 * 24 ** 2 / (image_array.shape[0]*image_array.shape[1]) #considerando q la imagen mide 24x24cm y la mostramos 16 veces mas grande

            segmentada[:, :, 3] = mascara_bin

            # Mostrar y guardar la imagen segmentada
            segmentada = Image.fromarray(segmentada, "RGBA")
            st.image(segmentada, caption="Imagen segmentada", use_column_width=True)
            st.info(f"El area segmentada mide aproximadamente: **{area:.0f} cm2**")
            segmentada.save(f"{path}_manualseg{ext}")

    elif option == "Clasificador Avanzado":
        handle = st.success(f"Procesando...")
        st.subheader("Clasificador de grado de avance de demencia con CNN", anchor= "prediction", divider= 'grey')
        advanced_model= build_and_load_model_from_state()
        img = cv2.imread(st.session_state.patient_data["image_file_path"], cv2.IMREAD_GRAYSCALE)
        inputs = pre_process_image([img])
        prediction = predict(advanced_model, inputs)[0]
        accuracy = PREDICTION_POWER[prediction]
        if 0.5 <= accuracy <0.7:
            color= "orange"
        elif accuracy >= 0.7 : 
            color= "green"
        elif accuracy < 0.5:
            color = "red"
        st.warning("La predicción se realiza por medio de una red neurnal convolucional del tipo ResNet 18 y debe ser utilizado con discresión y criterio de un profesional. Es solo a modo de guía.", icon="🚨")
        st.write(f"Según el clasificador, el paciente tiene un **caso de demencia**: **:{color}-background[{LABELS[prediction]}]**")
        st.write(f"**Dicho valor fue calculado con una precisión del: :{color}[{accuracy:.1f}%]** ")
        handle.empty()

def menu_page():

    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Inicio", "👥 Agregar Paciente", "🔎 Buscar Paciente", " 📊 Ver Resultados"])

    with tab1:
        tab1.header(":blue[¡Bienvenido!]👩🏻‍⚕️👨🏻‍⚕️")
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
            if name and lastname and validate_id(DNI) and validate_email(email) and doctor: #todos los datos completos
                # Guardamos datos del paciente
                alta_paciente(name, lastname, DNI, email, doctor)
                st.session_state.new_filename = get_patients_filename(lastname, DNI)
                st.session_state.patient_saved = True  # Marcamos que el paciente fue guardado
            elif not validate_id(DNI):
                st.error("El DNI debe ser un número entero con 6 o 7 cifras.")
            elif not validate_id(email):
                st.error("Ingrese una dirección de correo electrónica válida.")
            else:
                st.warning("Por favor, complete todos los datos antes de guardar.")

        # Solo mostramos el cargador de imágenes si el paciente ha sido guardado
        if st.session_state.patient_saved:
            subir_imagen(st.session_state.new_filename)

    with tab3:
        mostrar_paciente()

    with tab4:
        option = tab4.radio("Seleccioná una opción para proceder con el análisis de la imágen",
                            ("Segmentación Manual (draw)", "Segmentación Automática", "Clasificador Avanzado"))
        if st.session_state.patient_data["image_file_path"] is None:
            st.warning("Debe dar de alta un paciente y subir una imagen para proceder con el análisis")
        else:
            if tab4.button("Confirmar selección"):
                st.session_state.confirm = option
            if st.session_state.confirm == option:
                menu_results(st.session_state.confirm)
            else:
                st.session_state.confirm = None
                st.info("Confirme la selección para proceder.")

    st.sidebar.write(st.session_state.user_email)
    if st.sidebar.button("Cerrar Sesión"):
        st.session_state.logged_in = False
        st.rerun()

def main():
    if st.session_state.logged_in:
        menu_page()
    else:
        login_page()

main()
    