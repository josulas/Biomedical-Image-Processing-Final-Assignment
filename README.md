## Introduction

The current repository is aimed to host the final asignment developed as part of the Image Processing subject at Buenos Aires Technological Institute (ITBA). The objective of the project is to create a user-friendly medical tool to allow clinicians assess dementia progression through MRI scans. 

For that purpuse, we developed an application using Streamlit. Among its functionalities, users can:
- Log in and log out.
- Upload and delete patient data.
- Manually segment MRI scans.
- Automatically segment MRI scans with k-means and get meaningful metrics from the process.
- Employ a CNN classifier (ResNet-18) to predict dementia progression.

Images outputed by each process are automatically saved. For more information regarding the employed database and a teorethical context, refer to the [project's report](https://github.com/josulas/Biomedical-Image-Processing-Final-Assignment/blob/main/Informe%20Final.pdf). Notice the document is written in Spanish, as the university is located in a Spanish speaking country.

## Usage

It's advisable to execute the repository's code within a Windows machine, due to compatibility issues found employing a Mac OS computer. The project is intended to work in all three major platforms, but this feature has not been fully implemented yet. 

Python 3.12 is recomended as the interpreter version. The required packages are located at the requirements.txt file. After cloning the repository, we recomend to employ a virtual environment to install all the necessary dependencies.

Since no process has been created to add users, in order to start using the interface you'll have to manually add yourself to the dictionary of registered users in the main.py file located in the [interface folder](https://github.com/josulas/Biomedical-Image-Processing-Final-Assignment/tree/main/interface). Patients are managed using SQLite, meaning they are not harcoded in the application.

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


