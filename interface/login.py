import streamlit as st
import os
import sqlite3
import pandas as pd

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
    st.image('/Users/sofia/Downloads/mri.jpg', use_column_width=True) #nose como editar esto para que lo tengan toodos
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
    upload_folder = "images_database"
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Cargar la imagen aquí
    uploaded_file = st.file_uploader("Elija la imagen del paciente correspondiente:", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Extensión original del archivo
        extension = os.path.splitext(uploaded_file.name)[1]

        if filename:
            # Leer la imagen
            bytes_data = uploaded_file.getvalue()

            file_path = os.path.join(upload_folder, filename + extension) 
            with open(file_path, "wb") as f:
                f.write(bytes_data)

            st.write(f"🙌🏼 ✅ Puedes encontrar la imagen en: {file_path}")
            st.image(file_path)
        else:
            st.warning("Por favor, ingrese un nombre para la imagen.")


def alta_paciente(name, lastname, DNI, email,doctor):
    # Conectar a la base de datos (se creará automáticamente si no existe)
    conn = sqlite3.connect('pacientes.db')
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
    conn = sqlite3.connect('pacientes.db')
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



# Guardo datos del paciente para habilitar el resto
if 'paciente_guardado' not in st.session_state:
    st.session_state.paciente_guardado = False
    st.session_state.archivo = None

def menu_page():
    tab1, tab2, tab3, tab4 = st.tabs(["🏠 Inicio", "👥 Agregar Paciente", "🔎 Buscar Paciente", " 📊 Ver Resultados"])

    with tab1:
        tab1.subheader(f"Bienvenido Doctor!!👩🏻‍⚕️👨🏻‍⚕️")
        tab1.markdown("Elija la **opción correspondiente** para comenzar con las actividades")

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
            if name and lastname and DNI and email and doctor:
                # Guardamos datos del paciente
                alta_paciente(name, lastname, DNI, email, doctor)
                archivo = f"{name}_{lastname}"
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

if st.session_state.logged_in:
    menu_page()
else:
    login_page()