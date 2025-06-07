"""Neurologist interface for patient analysis."""


import base64
from io import BytesIO
import json
import time

import requests
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.api_client import APIClient
from libraries.segmentation.k_means import kmeans
from libraries.improving.filtering import conv2d


def gaussian_kernel_1d(size, sigma):
    """Generate 1D Gaussian kernel using NumPy."""
    kernel = np.zeros(size)
    center = size // 2
    for i in range(size):
        x = i - center
        kernel[i] = np.exp(-(x * x) / (2 * sigma * sigma))
    # Normalize
    kernel = kernel / np.sum(kernel)
    return kernel.reshape(-1, 1)

class NeurologistInterface:
    """Interface for neurologist users."""
    def __init__(self, api_client: APIClient,
                 model_server_url: str):
        self.api_client = api_client
        self.model_server_url = model_server_url

    def render(self, page: str):
        """Render neurologist interface based on current page."""
        if page == "neurologist_patients":
            self.show_patients()
        elif page == "neurologist_analysis":
            self.show_analysis_tools()
        elif page == "neurologist_assignments":
            self.show_assignments()
        else:
            st.error("Unknown page")

    def show_patients(self):
        """Show assigned patients and their images."""
        st.subheader("👥 My Assigned Patients")
        # Get assigned patients
        patients_result = self.api_client.get_patients()
        if patients_result["success"]:
            patients = patients_result["data"]
            if not patients:
                st.info("You haven't been assigned any patients yet.\
                        Contact your administrator.")
                return
            # Patient selection
            patient_options = \
                {f"{p['first_name']} {p['last_name']} ({p['email']})": p for p in patients}
            selected_patient_name = st.selectbox("Select a patient:", list(patient_options.keys()))
            if selected_patient_name:
                selected_patient = patient_options[selected_patient_name]
                st.session_state.selected_patient = selected_patient
                # Show patient info
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### Patient Information")
                    st.write(f"**Name:** {selected_patient['first_name']} \
                             {selected_patient['last_name']}")
                    st.write(f"**Email:** {selected_patient['email']}")
                    if selected_patient.get('phone'):
                        st.write(f"**Phone:** {selected_patient['phone']}")
                    if selected_patient.get('date_of_birth'):
                        st.write(f"**Date of Birth:** {selected_patient['date_of_birth']}")
                with col2:
                    st.markdown("### Quick Actions")
                    if st.button("🔬 Analyze Images", use_container_width=True, type="primary"):
                        st.session_state.current_page = "neurologist_analysis"
                        st.rerun()
                # Get patient's images and results
                self.show_patient_images_and_results(selected_patient['id'])
        else:
            st.error(f"Failed to load patients: {patients_result.get('error', 'Unknown error')}")

    def show_patient_images_and_results(self, patient_id):
        """Show patient's images and existing results."""
        st.divider()
        # Get study results for this patient
        results_response = self.api_client.get_study_results(patient_id)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 Completed Studies")
            if results_response["success"]:
                results = results_response["data"]
                if results:
                    for result in results:
                        with \
                            st.expander(f"Study: {result['study_name'] \
                                or result['image_filename']} - {result['created_at'][:10]}"):
                            st.write(f"**Classification:** {result['classification_result']}")
                            st.write(f"**Confidence:** {result['confidence_score']:.2f}%")
                            if result['notes']:
                                st.write(f"**Notes:** {result['notes']}")
                            st.write(f"**Date:** {result['created_at']}")
                else:
                    st.info("No completed studies for this patient.")
            else:
                st.error("Failed to load study results.")
        with col2:
            st.markdown("### 📤 Pending Analysis")
            # This would show images without results - we'll implement this in the analysis section
            st.info("Use the 'Analyze Images' button to process pending studies.")

    def show_pending_analysis_summary(self, patient_id):
        """Show summary of pending analysis for a patient."""
        images_response = self.get_patient_images(patient_id)
        results_response = self.api_client.get_study_results(patient_id)
        if images_response["success"] and results_response["success"]:
            images = images_response["data"]
            results = results_response["data"]
            # Get analyzed image IDs
            analyzed_image_ids = {result['image_id'] for result in results}
            # Count pending images
            pending_count = len([img for img in images if img['id'] not in analyzed_image_ids])
            return pending_count, len(images)
        return 0, 0

    def show_assignments(self):
        """Show all assignments overview."""
        st.subheader("📋 Assignment Overview")
        # Get assigned patients
        patients_result = self.api_client.get_patients()
        if patients_result["success"]:
            patients = patients_result["data"]
            if not patients:
                st.info("You haven't been assigned any patients yet.")
                return
            # Statistics
            col1, col2, col3 = st.columns(3)
            total_patients = len(patients)
            with col1:
                st.metric("Assigned Patients", total_patients)
            # Get total studies and pending studies
            total_studies = 0
            completed_studies = 0
            pending_studies = 0
            for patient in patients:
                results_response = self.api_client.get_study_results(patient['id'])
                if results_response["success"]:
                    completed_studies += len(results_response["data"])
                # Get pending count for this patient
                pending_count, total_count = self.show_pending_analysis_summary(patient['id'])
                total_studies += total_count
                pending_studies += pending_count
            with col2:
                st.metric("Completed Studies", completed_studies)
            with col3:
                st.metric("Pending Studies", pending_studies)
            st.divider()
            # Patients table
            st.markdown("### Patient List")
            for patient in patients:
                with st.expander(f"👤 {patient['first_name']} {patient['last_name']}"):
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.write(f"**Email:** {patient['email']}")
                        if patient.get('phone'):
                            st.write(f"**Phone:** {patient['phone']}")
                        # Show recent activity
                        results_response = self.api_client.get_study_results(patient['id'])
                        if results_response["success"] and results_response["data"]:
                            latest_result = results_response["data"][0]
                            st.write(f"**Last Study:** {latest_result['created_at'][:10]}")
                        else:
                            st.write("**Status:** No studies completed")
                    with col_b:
                        if st.button(f"Analyze {patient['first_name']}", key=f"analyze_{patient['id']}"):
                            st.session_state.selected_patient = patient
                            st.session_state.current_page = "neurologist_analysis"
                            st.rerun()
        else:
            st.error(f"Failed to load assignments: {patients_result.get('error', 'Unknown error')}")

    def show_analysis_tools(self):
        """Show image analysis tools."""
        st.subheader("🔬 Image Analysis Tools")
        # Check if we have a selected patient
        if 'selected_patient' not in st.session_state:
            st.warning("Please select a patient first from the Patients tab.")
            if st.button("Go to Patients"):
                st.session_state.current_page = "neurologist_patients"
                st.rerun()
            return
        patient = st.session_state.selected_patient
        st.info(f"Analyzing images for: **{patient['first_name']} {patient['last_name']}**")        
        # Get patient's images (we need to add this endpoint)
        images_result = self.get_patient_images(patient['id'])
        if not images_result["success"]:
            st.error(f"Failed to load patient images: \
                     {images_result.get('error', 'Unknown error')}")
            return
        images = images_result["data"]
        if not images:
            st.warning("This patient has no uploaded images.")
            return
        # Image selection
        selected_image = None
        if 'selected_pending_image' in st.session_state:
            selected_image = st.session_state.selected_pending_image
            st.success(f"📷 **Selected for analysis:** {selected_image['study_name'] or selected_image['original_filename']}")
            # Clear the pending selection
            del st.session_state.selected_pending_image
        else:
            # Regular image selection
            image_options = {f"{img['study_name'] or img['original_filename']} (Uploaded: {img['uploaded_at'][:10]})": img for img in images}
            selected_image_name = st.selectbox("Select an image to analyze:", list(image_options.keys()))
            if selected_image_name:
                selected_image = image_options[selected_image_name]
        if selected_image:
            # Check if this image already has results
            existing_result = self.check_existing_results(selected_image['id'])
            if existing_result:
                st.success("✅ This image has already been analyzed.")
                with st.expander("View Existing Results"):
                    st.write(f"**Classification:** {existing_result['classification_result']}")
                    st.write(f"**Confidence:** {existing_result['confidence_score']:.2f}%")
                    if existing_result['notes']:
                        st.write(f"**Notes:** {existing_result['notes']}")
                    st.write(f"**Analysis Date:** {existing_result['created_at']}")
                
                if st.button("🔄 Re-analyze Image"):
                    self.perform_image_analysis(selected_image, patient)
            else:
                st.info("This image is pending analysis.")
                self.perform_image_analysis(selected_image, patient)

    def get_patient_images(self, patient_id):
        """Get images for a specific patient (this needs to be added to backend)."""
        try:
            # For now, we'll use a placeholder - this needs to be implemented in backend
            response = self.api_client.session.get(f"{self.api_client.base_url}/patients/{patient_id}/images")
            response.raise_for_status()
            return {"success": True, "data": response.json()}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def check_existing_results(self, image_id):
        """Check if image already has results."""
        try:
            response = self.api_client.session.get(f"{self.api_client.base_url}/images/{image_id}/results")
            if response.status_code == 200:
                return response.json()
            return None
        except Exception:
            return None

    def perform_image_analysis(self, image, patient):
        """Perform comprehensive image analysis."""
        st.divider()
        st.markdown("### 🖼️ Image Analysis Interface")
        # Load and display image
        image_data = self.load_image(image['id'])
        if image_data is None:
            st.error("Failed to load image.")
            return
        # Display original image
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("#### Original Image")
            st.image(image_data, caption=f"{image['study_name'] or image['original_filename']}",  use_container_width=True)
        with col2:
            st.markdown("#### Analysis Tools")
            # Tool selection
            analysis_options = st.multiselect(
                "Select analysis tools to apply:",
                ["🤖 AI Classification", "🎯 Automatic Segmentation", "✏️ Manual Segmentation", "📊 Quantitative Analysis"],
                default=["🤖 AI Classification"]
            )
        # Analysis results storage
        analysis_results = {}
        st.divider()
        # Apply selected tools
        if "🤖 AI Classification" in analysis_options:
            classification_result = self.apply_ai_classification(image_data)
            analysis_results['classification'] = classification_result
        if "🎯 Automatic Segmentation" in analysis_options:
            segmentation_result = self.apply_automatic_segmentation(image_data)
            analysis_results['auto_segmentation'] = segmentation_result
        if "✏️ Manual Segmentation" in analysis_options:
            manual_seg_result = self.apply_manual_segmentation(image_data)
            analysis_results['manual_segmentation'] = manual_seg_result
        if "📊 Quantitative Analysis" in analysis_options:
            quant_result = self.apply_quantitative_analysis(image_data, analysis_results.get('auto_segmentation'))
            analysis_results['quantitative'] = quant_result
        # Final diagnosis section
        st.divider()
        st.markdown("### 📋 Final Diagnosis")
        self.create_final_diagnosis_form(image, patient, analysis_results)
    
    def load_image(self, image_id):
        """Load image from backend."""
        try:
            response = self.api_client.session.get(f"{self.api_client.base_url}/images/{image_id}/download")
            if response.status_code == 200:
                return Image.open(BytesIO(response.content))
            return None
        except Exception:
            # For demo purposes, return a placeholder
            st.warning("Using demo image - image loading endpoint needs implementation")
            return None

    def apply_ai_classification(self, image_data):
        """Apply AI classification using the model service."""
        st.markdown("#### 🤖 AI Classification Results")
        with st.spinner("Running AI classification..."):
            try:
                if image_data is None:
                    st.error("No image data available")
                    return None
                if image_data.mode != 'RGB':
                    image_data = image_data.convert('RGB')
                # Convert PIL image to base64
                buffered = BytesIO()
                image_data.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode()
                # Call actual model service
                response = requests.post(
                    f"{self.model_server_url}/predict",
                    json={"image": img_base64},
                    timeout=30
                )
                if response.status_code == 200:
                    result = response.json()
                    col1, col2 = st.columns(2)  # Changed from 3 columns to 2
                    with col1:
                        st.metric("Classification", result["class_name"])
                    with col2:
                        st.metric("Confidence", f"{result['confidence']:.1f}%")
                    st.success("AI Classification completed!")
                    return result
                else:
                    st.error(f"Model service error: {response.text}")
                    return None
            except requests.exceptions.RequestException as e:
                st.error(f"Failed to connect to model service: {str(e)}")
                return None
            except Exception as e:
                st.error(f"AI Classification failed: {str(e)}")
                return None

    def apply_automatic_segmentation(self, image_data):
        """Apply automatic K-means segmentation."""
        st.markdown("#### 🎯 Automatic Segmentation (K-means)")
        with st.spinner("Performing automatic segmentation..."):
            try:
                if image_data is None:
                    st.error("No image data available")
                    return None
                img_array = np.array(image_data.convert('L'))  # Convert to grayscale
                # Apply Gaussian filter (from old_app.py)
                epsilon = 0.1
                sigma = 3
                gaussianDim = int(np.ceil(np.sqrt(-2 * sigma ** 2 * np.log(epsilon * sigma * np.sqrt(2 * np.pi)))))
                gaussianKernel1D = gaussian_kernel_1d(gaussianDim, sigma)
                gaussianKernel = np.outer(gaussianKernel1D, gaussianKernel1D)
                filtered_image = conv2d(img_array, gaussianKernel)
                # Apply K-means segmentation
                _, labels, centers = kmeans(filtered_image.flatten(), 3, attempts=5)
                centers = centers.astype(np.uint8)
                segmented_kmeans = centers[labels].reshape(filtered_image.shape)
                # Identify tissue types based on intensity
                sorted_centers = sorted(centers)
                background_idx = np.argmax(centers == sorted_centers[0])  # Darkest
                grey_matter_idx = np.argmax(centers == sorted_centers[1])  # Medium
                white_matter_idx = np.argmax(centers == sorted_centers[2])  # Brightest
                # Create binary masks
                segmented_white_matter = np.where(segmented_kmeans == centers[white_matter_idx], 1, 0)
                segmented_grey_matter = np.where(segmented_kmeans == centers[grey_matter_idx], 1, 0)
                # segmented_background = np.where(segmented_kmeans == centers[background_idx], 1, 0)
                # Calculate ratios (from old_app.py)
                grey_count = np.sum(segmented_grey_matter)
                white_count = np.sum(segmented_white_matter)
                total_brain = grey_count + white_count
                if white_count > 0:
                    gray_white_ratio = grey_count / white_count
                    gray_percentage = (grey_count * 100) / total_brain
                    # Assuming 24x24 cm image size as in old_app
                    total_area_cm2 = total_brain / (img_array.shape[0] * img_array.shape[1]) * 24 ** 2
                else:
                    gray_white_ratio = 0
                    gray_percentage = 0
                    total_area_cm2 = 0
                # Display results
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Gray Matter Segmentation**")
                    fig_gray, ax_gray = plt.subplots(figsize=(6, 4))
                    ax_gray.imshow(segmented_grey_matter * 255, cmap='gray')
                    ax_gray.axis('off')
                    ax_gray.set_title('Gray Matter')
                    st.pyplot(fig_gray)
                    plt.close(fig_gray)
                with col2:
                    st.markdown("**White Matter Segmentation**")
                    fig_white, ax_white = plt.subplots(figsize=(6, 4))
                    ax_white.imshow(segmented_white_matter * 255, cmap='gray')
                    ax_white.axis('off')
                    ax_white.set_title('White Matter')
                    st.pyplot(fig_white)
                    plt.close(fig_white)
                # Show complete segmentation
                st.markdown("**Complete Segmentation**")
                fig_complete, ax_complete = plt.subplots(figsize=(8, 6))
                ax_complete.imshow(segmented_kmeans, cmap='gray')
                ax_complete.axis('off')
                ax_complete.set_title('K-means Segmentation (3 clusters)')
                st.pyplot(fig_complete)
                plt.close(fig_complete)
                # Quantitative results
                ratio_result = {
                    "gray_white_ratio": round(gray_white_ratio, 3),
                    "gray_percentage": round(gray_percentage, 2),
                    "total_area_cm2": round(total_area_cm2, 2),
                    "white_matter_pixels": int(white_count),
                    "gray_matter_pixels": int(grey_count)
                }
                st.markdown("**Quantitative Results:**")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Gray/White Ratio", f"{ratio_result['gray_white_ratio']:.3f}")
                with col_b:
                    st.metric("Gray Matter %", f"{ratio_result['gray_percentage']:.1f}%")
                with col_c:
                    st.metric("Total Area", f"{ratio_result['total_area_cm2']:.1f} cm²")
                # Clinical interpretation (from old_app.py knowledge)
                if ratio_result['gray_white_ratio'] < 0.6:
                    st.warning("⚠️ Low gray/white matter ratio - may indicate significant atrophy")
                elif ratio_result['gray_white_ratio'] < 0.65:
                    st.info("ℹ️ Slightly reduced gray/white matter ratio")
                else:
                    st.success("✅ Gray/white matter ratio within normal range")
                st.success("Automatic segmentation completed!")
                return ratio_result
            except Exception as e:
                st.error(f"Automatic segmentation failed: {str(e)}")
                return None

    def apply_manual_segmentation(self, image_data):
        """Apply manual segmentation using drawing tools."""
        st.markdown("#### ✏️ Manual Segmentation")
        if image_data is None:
            st.error("No image data available")
            return None
        # Resize image for better drawing experience (from old_app.py)
        img_array = np.array(image_data)
        # Convert to RGB if it's not already (remove alpha channel if present)
        if img_array.shape[-1] == 4:  # RGBA
            img_array = img_array[:, :, :3]  # Remove alpha channel
        elif len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
        # Create PIL image from the cleaned array
        clean_image = Image.fromarray(img_array.astype(np.uint8))
        # Resize image for better drawing experience (from old_app.py)
        original_size = clean_image.size
        scaled_image = clean_image.resize((
            min(800, 4 * original_size[0]),  # Limit max width to 800px
            min(600, 4 * original_size[1])   # Limit max height to 600px
        ), Image.Resampling.LANCZOS)
        st.info("Draw on the image to mark regions of interest. The area will be calculated automatically.")
        # Drawing controls
        drawing_mode = st.selectbox("Drawing Mode:",
                                    ["Region of Interest",
                                     "Gray Matter",
                                     "White Matter",
                                     "Lesions"])
        brush_size = st.slider("Brush Size:", 3, 50, 20)
        st.markdown("**Drawing Canvas**")
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=brush_size,
            stroke_color="red",
            background_image=scaled_image,
            height=scaled_image.height,
            width=scaled_image.width,
            drawing_mode="freedraw",
            key="manual_segmentation_canvas"
        )
        if canvas_result.image_data is not None:
            # Extract the drawn mask
            mask = np.array(canvas_result.image_data[:, :, 3])  # Alpha channel
            mask_binary = np.where(mask > 0, 255, 0).astype("uint8")
            # Calculate area (from old_app.py formula)
            drawn_pixels = np.sum(mask_binary) / 255
            # Assuming 24x24 cm image size, scaled by 4x4 = 16 times larger
            total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
            area_cm2 = drawn_pixels * 24 ** 2 / total_pixels
            if drawn_pixels > 0:
                # Show the segmented region
                segmented_image = np.array(scaled_image.convert("RGBA"))
                segmented_image[:, :, 3] = mask_binary
                segmented_result = Image.fromarray(segmented_image, "RGBA")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Original with Overlay**")
                    st.image(canvas_result.image_data,
                             caption="Manual Segmentation",
                              use_container_width=True)
                with col2:
                    st.markdown("**Segmented Region**")
                    st.image(segmented_result,
                             caption="Extracted Region",
                              use_container_width=True)
                st.success(f"**Segmented area: {area_cm2:.1f} cm²**")
                return {
                    "mode": drawing_mode,
                    "area_cm2": round(area_cm2, 2),
                    "pixels_drawn": int(drawn_pixels),
                    "mask": mask_binary
                }
            else:
                st.info("Draw on the image to segment regions.")
                return None
        return None

    def apply_quantitative_analysis(self, image_data, segmentation_data):
        """Apply quantitative analysis based on segmentation."""
        st.markdown("#### 📊 Quantitative Analysis")
        if not segmentation_data:
            st.warning("Automatic segmentation required for quantitative analysis.")
            return None
        if image_data is None:
            st.error("No image data available")
            return None
        with st.spinner("Calculating quantitative metrics..."):
            try:
                # Use segmentation data for calculations
                gray_pixels = segmentation_data.get('gray_matter_pixels', 0)
                white_pixels = segmentation_data.get('white_matter_pixels', 0)
                total_brain_pixels = gray_pixels + white_pixels
                # Image dimensions for area calculations
                img_array = np.array(image_data.convert('L'))
                total_image_pixels = img_array.shape[0] * img_array.shape[1]
                # Volume estimations (assuming slice thickness and pixel spacing)
                # These are estimates - in real clinical setting, DICOM headers would provide real measurements
                pixel_size_mm = 0.5  # Assume 0.5mm pixel size
                slice_thickness_mm = 5.0  # Assume 5mm slice thickness
                # Convert pixels to volume (mm³, then cm³)
                total_brain_volume_mm3 = \
                    total_brain_pixels * (pixel_size_mm ** 2) * slice_thickness_mm
                gray_matter_volume_mm3 = \
                    gray_pixels * (pixel_size_mm ** 2) * slice_thickness_mm
                white_matter_volume_mm3 = \
                    white_pixels * (pixel_size_mm ** 2) * slice_thickness_mm
                # Convert to cm³
                total_brain_volume = total_brain_volume_mm3 / 1000
                gray_matter_volume = gray_matter_volume_mm3 / 1000
                white_matter_volume = white_matter_volume_mm3 / 1000
                # Estimate ventricle volume (assuming background pixels near center are ventricles)
                # This is a simplified estimation
                center_region = img_array[img_array.shape[0]//3:2*img_array.shape[0]//3,                                        img_array.shape[1]//3:2*img_array.shape[1]//3]
                dark_pixels = np.sum(center_region < 50)  # Very dark pixels likely ventricles
                ventricle_volume = \
                    dark_pixels * (pixel_size_mm ** 2) * slice_thickness_mm / 1000
                # Calculate atrophy index (simplified)
                # Normal brain volume for reference (approximate)
                normal_total_volume = 1400  # cm³ approximate normal brain volume
                atrophy_index = max(0, (normal_total_volume - total_brain_volume) / normal_total_volume)
                metrics = {
                    "total_brain_volume": round(total_brain_volume, 1),
                    "gray_matter_volume": round(gray_matter_volume, 1),
                    "white_matter_volume": round(white_matter_volume, 1),
                    "ventricle_volume": round(ventricle_volume, 1),
                    "atrophy_index": round(atrophy_index, 3),
                    "brain_to_image_ratio": round(total_brain_pixels / total_image_pixels, 3)
                }
                st.success("Quantitative analysis completed!")
                # Display metrics in a grid
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Volume Measurements (cm³):**")
                    st.metric("Total Brain Volume", f"{metrics['total_brain_volume']:.1f}")
                    st.metric("Gray Matter Volume", f"{metrics['gray_matter_volume']:.1f}")
                    st.metric("White Matter Volume", f"{metrics['white_matter_volume']:.1f}")
                with col2:
                    st.markdown("**Clinical Indices:**")
                    st.metric("Ventricle Volume", f"{metrics['ventricle_volume']:.1f}")
                    st.metric("Atrophy Index", f"{metrics['atrophy_index']:.3f}")
                    st.metric("Brain/Image Ratio", f"{metrics['brain_to_image_ratio']:.3f}")
                # Clinical interpretation
                if metrics['atrophy_index'] > 0.3:
                    st.error("🔴 Significant atrophy detected")
                elif metrics['atrophy_index'] > 0.15:
                    st.warning("🟡 Mild to moderate atrophy present")
                else:
                    st.success("🟢 Brain volume within normal range")
                # Add disclaimer
                st.info("⚠️ Volume calculations are estimates based on pixel analysis. \
                        Clinical interpretation should consider DICOM metadata and other clinical factors.")
                return metrics
                
            except Exception as e:
                st.error(f"Quantitative analysis failed: {str(e)}")
                return None

    def create_final_diagnosis_form(self, image, patient, analysis_results):
        """Create final diagnosis form."""
        with st.form("diagnosis_form"):
            st.markdown("**Final Clinical Assessment**")
            
            # Get classification options from model service
            try:
                response = requests.get(f"{self.model_server_url}/class-names", timeout=10)
                if response.status_code == 200:
                    labels_data = response.json()
                    # Expected format: {"class_names": {0: "Mild_Demented", 1: "Moderate_Demented", ...}}
                    classification_options = labels_data.get("class_names", {})
                    # Convert string keys to integers if needed
                    if isinstance(list(classification_options.keys())[0], str):
                        classification_options = {int(k): v for k, v in classification_options.items()}
                else:
                    # Fallback - use the correct labels from model service
                    classification_options = {
                        0: "Mild_Demented",
                        1: "Moderate_Demented",
                        2: "Non_Demented",
                        3: "Very_Mild_Demented"
                    }
            except Exception:
                # Fallback if model service is unavailable
                classification_options = {
                    0: "Mild_Demented",
                    1: "Moderate_Demented",
                    2: "Non_Demented",
                    3: "Very_Mild_Demented"
                }
            
            # Default to AI classification if available
            default_classification = 0
            if 'classification' in analysis_results and analysis_results['classification']:
                default_classification = analysis_results['classification']['prediction']
            
            final_classification = st.selectbox(
                "Final Classification:",
                options=list(classification_options.keys()),
                format_func=lambda x: classification_options[x].replace('_', ' '),  # Make labels more readable
                index=default_classification
            )
            
            # Confidence assessment
            confidence_score = st.slider(
                "Diagnostic Confidence (%):",
                0, 100, 
                int(analysis_results.get('classification', {}).get('confidence', 80))
            )
            
            # Clinical notes
            clinical_notes = st.text_area(
                "Clinical Notes and Observations:",
                placeholder="Enter your clinical observations, analysis summary, and recommendations...",
                height=150
            )
            
            # Analysis summary
            if analysis_results:
                st.markdown("**Analysis Summary:**")
                if 'classification' in analysis_results:
                    ai_result = analysis_results['classification']
                    readable_class_name = ai_result['class_name'].replace('_', ' ')
                    st.write(f"• AI Classification: {readable_class_name} ({ai_result['confidence']:.1f}% confidence)")
                if 'auto_segmentation' in analysis_results:
                    seg_result = analysis_results['auto_segmentation']
                    st.write(f"• Gray/White Ratio: {seg_result['gray_white_ratio']:.3f}")
                    st.write(f"• Gray Matter: {seg_result['gray_percentage']:.1f}%")
                if 'quantitative' in analysis_results:
                    quant_result = analysis_results['quantitative']
                    st.write(f"• Atrophy Index: {quant_result['atrophy_index']:.3f}")
            
            # Submit buttons
            col1, col2 = st.columns([1, 1])
            
            with col1:
                submit_diagnosis = st.form_submit_button(
                    "💾 Submit Final Diagnosis",
                    type="primary",
                    use_container_width=True
                )
            
            with col2:
                save_draft = st.form_submit_button(
                    "📝 Save as Draft",
                    use_container_width=True
                )
            
            if submit_diagnosis:
                self.save_study_result(
                    image_id=image['id'],
                    patient_id=patient['id'],
                    classification=final_classification,
                    confidence=confidence_score,
                    notes=clinical_notes,
                    analysis_data=analysis_results,
                    is_final=True
                )
            elif save_draft:
                self.save_study_result(
                    image_id=image['id'],
                    patient_id=patient['id'],
                    classification=final_classification,
                    confidence=confidence_score,
                    notes=clinical_notes,
                    analysis_data=analysis_results,
                    is_final=False
                )

    def save_study_result(self,
                          image_id,
                          patient_id,
                          classification,
                          confidence,
                          notes,
                          analysis_data,
                          is_final=True):
        """Save study result to backend."""
        try:
            result_data = {
                "image_id": image_id,
                "patient_id": patient_id,
                "classification_result": classification,
                "confidence_score": confidence,
                "notes": notes,
                "segmentation_data": json.dumps(analysis_data) \
                    if analysis_data else None
            }
            response = self.api_client.session.post(
                f"{self.api_client.base_url}/studies/results",
                json=result_data
            )
            if response.status_code == 200:
                if is_final:
                    st.success("✅ Final diagnosis submitted successfully!")
                    st.balloons()
                    time.sleep(2)
                else:
                    st.success("📝 Draft saved successfully!")
                # Clear session state and redirect
                if 'selected_patient' in st.session_state:
                    del st.session_state['selected_patient']
                st.session_state.current_page = "neurologist_patients"
                st.rerun()
            else:
                st.error(f"Failed to save result: {response.text}")
        except Exception as e:
            st.error(f"Error saving result: {str(e)}")
