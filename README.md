# Biomedical Image Processing - Dementia Diagnosis Toolkit

## Introduction

The current repository hosts the final assignment developed as part of the Image Processing subject at Buenos Aires Technological Institute (ITBA). The objective is to create a professional medical tool that allows neurologists and healthcare professionals to assess dementia progression through MRI scans using advanced image processing and AI techniques.

This is a **microservices-based web application** built with modern technologies including FastAPI, Streamlit, and Docker, providing a scalable and user-friendly platform for medical image analysis.

## 🏗️ Architecture

The application consists of three main microservices:

- **🖥️ Frontend Service** (Streamlit) - User interface for neurologists, patients, and administrators
- **⚙️ Backend Service** (FastAPI) - User management, authentication, and data persistence
- **🤖 Model Service** (FastAPI) - AI classification and image processing workflows

### Key Features

#### For Neurologists:
- 🔍 **AI-Powered Classification** - ResNet-18 CNN for automated dementia severity assessment
- 🎯 **Automatic Segmentation** - K-means clustering for gray/white matter analysis
- ✏️ **Manual Segmentation** - Interactive drawing tools for region-of-interest marking
- 📊 **Quantitative Analysis** - Volume calculations and atrophy indices
- 📝 **Clinical Reporting** - Comprehensive diagnostic reports with confidence metrics

#### For Patients:
- 📤 **Image Upload** - Secure MRI scan submission
- 📋 **Study History** - View past analyses and results
- 👩‍⚕️ **Doctor Assignment** - Connect with assigned neurologists

#### For Administrators:
- 👥 **User Management** - Create and manage neurologist and patient accounts
- 🔗 **Assignment Management** - Assign patients to neurologists
- 📈 **System Analytics** - Monitor usage and system health

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- Git

### Installation & Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/josulas/Biomedical-Image-Processing-Final-Assignment.git
   cd Biomedical-Image-Processing-Final-Assignment
   ```

2. **Configure environment variables:**
  
Following the .env.example files in each service folder, create `.env` files with the necessary configurations.
Its only necessary to modify the access settings for the backend file. If the model won't be retrained, the rest of the .env files can be left as they come from the .example files.

   - **Frontend**: `frontend/.env`
   - **Backend**: `backend/.env`
   - **Model Service**: `model_service/.env`

3. **Build and run the application:**
   ```bash
   docker-compose up --build
   ```

4. **Access the application:**
    - Frontend: [http://localhost:8501](http://localhost:8501)

5. **Development Mode:**

  ```bash
  docker-compose -f compose.yml -f compose-dev.yml up --build --watch
  ```

It also exposes the following additional ports:
- Backend: `8000`
- Model Service: `8001`

## Autores

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="33%"><a href="https://github.com/sofiabouzo"><img src="https://avatars.githubusercontent.com/u/180412392?v=4" width="150px;" alt="Sofía Bouzo"/><br /><sub><b>Sofía Bouzo</b></sub></a><br /></td>
      <td align="center" valign="top" width="33%"><a href="https://github.com/fernandezmarti"><img src="https://avatars.githubusercontent.com/u/163201630?v=4" width="150px;" alt="Martín Fernández"/><br /><sub><b>Martín Fernández </b></sub></a><br /></td>
      <td align="center" valign="top" width="33%"><a href="https://github.com/josulas"><img src="https://avatars.githubusercontent.com/u/89985451?v=4" width="150px;" alt="Josue F. Laszeski"/><br /><sub><b>Josue F. Laszeski</b></sub></a><br /></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

## Aknowledgments

We want to specially thanks the course professors and helpers, for their constant support in problem resolution and availability to answer our inquiries. 


