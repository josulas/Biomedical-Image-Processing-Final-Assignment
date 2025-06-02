"""Patient interface for image upload and results viewing."""


from typing import Dict

import streamlit as st


class PatientInterface:
    """Interface for patient users."""
    
    def __init__(self, api_client):
        self.api_client = api_client
    
    def render(self, page: str):
        """Render patient interface based on current page."""
        if page == "patient_upload":
            self.show_upload_interface()
        elif page == "patient_results":
            self.show_results()
        elif page == "patient_manage":  # Add this new page
            self.show_manage_images()
        else:
            raise ValueError(f"Unknown page: {page}")
    
    def show_upload_interface(self):
        """Show image upload interface."""
        st.subheader("📤 Upload Medical Images")
        st.info("Upload your medical images for neurological evaluation and analysis.")
        
        # Load available neurologists
        neurologists_result = self.api_client.get_neurologists()
        if not neurologists_result["success"]:
            st.error(f"❌ Failed to load neurologists: {neurologists_result.get('error', 'Unknown error')}")
            return
        
        neurologists = neurologists_result["data"]
        if not neurologists:
            st.warning("⚠️ No neurologists available for assignment at this time.")
            return
        
        with st.form("upload_image_form", clear_on_submit=True):
            st.markdown("#### Image Upload Details")
            
            col1, col2 = st.columns(2)
            
            with col1:
                uploaded_file = st.file_uploader(
                    "Select Medical Image",
                    type=["jpg", "jpeg", "png", "tiff", "dicom", "dcm"],
                    help="Supported formats: JPG, PNG, TIFF, DICOM"
                )
                
                study_name = st.text_input(
                    "Study Name",
                    placeholder="e.g., MRI Brain Scan, CT Head, etc.",
                    help="Descriptive name for this medical study"
                )
            
            with col2:
                # Neurologist selection
                neurologist_options = [f"Dr. {n['first_name']} {n['last_name']}" for n in neurologists]
                selected_neurologist_idx = st.selectbox(
                    "Assign to Neurologist",
                    range(len(neurologist_options)),
                    format_func=lambda x: neurologist_options[x],
                    help="Select the neurologist who will evaluate this study"
                )
                
                # Additional notes
                notes = st.text_area(
                    "Additional Notes (Optional)",
                    placeholder="Any relevant medical history or symptoms...",
                    height=100,
                    help="Provide any context that might help with the evaluation"
                )
            
            # Upload button
            submit_upload = st.form_submit_button(
                "📤 Upload Image",
                use_container_width=True,
                type="primary"
            )
            
            if submit_upload:
                if uploaded_file and study_name:
                    selected_neurologist = neurologists[selected_neurologist_idx]
                    
                    with st.spinner("Uploading image and creating assignment..."):
                        # Upload image with study name
                        file_data = (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                        upload_result = self.api_client.upload_image(file_data, study_name)
                        
                        if upload_result["success"]:
                            # Create assignment
                            assignment_result = self.api_client.assign_to_neurologist(selected_neurologist['id'])
                            
                            if assignment_result["success"]:
                                st.success(f"✅ Image uploaded successfully and assigned to Dr. {selected_neurologist['first_name']} {selected_neurologist['last_name']}!")
                                st.balloons()
                                st.rerun()
                            else:
                                st.warning(f"⚠️ Image uploaded but assignment failed: {assignment_result.get('error', 'Unknown error')}")
                        else:
                            st.error(f"❌ Upload failed: {upload_result.get('error', 'Unknown error')}")
                else:
                    st.warning("⚠️ Please select an image and provide a study name")
        
        # Show recent uploads
        st.divider()
        st.markdown("#### Recent Uploads")
        self._show_recent_uploads()
    
    def _show_recent_uploads(self, limit: int = 5):
        """Show recent image uploads."""
        images_result = self.api_client.get_my_images()
        
        if not images_result["success"]:
            st.error(f"Failed to load images: {images_result.get('error', 'Unknown error')}")
            return
        
        images = images_result["data"][:limit]  # Show only recent uploads
        
        if not images:
            st.info("No images uploaded yet.")
            return
        
        for image in images:
            with st.expander(f"📁 {image['study_name'] or 'Unnamed Study'} - {image['uploaded_at'][:10]}"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**File:** {image['original_filename']}")
                    st.write(f"**Study:** {image['study_name'] or 'Not specified'}")
                    st.write(f"**Uploaded:** {image['uploaded_at'][:16]}")
                
                with col2:
                    st.info("💡 Use Dashboard to manage images")
    
    def show_results(self):
        """Show patient results."""
        st.subheader("📊 My Study Results")
        st.info("View completed neurological evaluations and analysis results.")
        
        # Load results
        results_result = self.api_client.get_my_results()
        
        if not results_result["success"]:
            st.error(f"❌ Failed to load results: {results_result.get('error', 'Unknown error')}")
            return
        
        results = results_result["data"]
        
        if not results:
            st.info("🔬 No completed studies yet. Results will appear here once your neurologist completes the analysis.")
            return
        
        # Group results by study/image
        for result in results:
            with st.expander(f"📋 {result['study_name'] or 'Study'} - Completed {result['created_at'][:10]}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Study Information**")
                    st.write(f"📁 **File:** {result['image_filename']}")
                    st.write(f"📅 **Completed:** {result['created_at'][:16]}")
                    st.write(f"👨‍⚕️ **Evaluated by:** Dr. {result['neurologist_name']}")
                
                with col2:
                    st.markdown("**Clinical Assessment**")
                    # Only show patient-relevant information, hide AI technical details
                    if result['classification_result'] is not None:
                        classification_text = "Positive findings detected" if result['classification_result'] == 1 else "No significant findings"
                        st.write(f"🔍 **Assessment:** {classification_text}")
                    
                    if result['confidence_score'] is not None:
                        # Convert technical confidence to patient-friendly language
                        confidence_level = "High" if result['confidence_score'] > 0.8 else "Moderate" if result['confidence_score'] > 0.6 else "Low"
                        st.write(f"📊 **Confidence Level:** {confidence_level}")
                
                # Clinical notes (always show if present)
                if result['notes']:
                    st.markdown("**Neurologist's Notes**")
                    st.write(result['notes'])
                
                # Download/view options
                col_a, col_b = st.columns(2)
                with col_a:
                    if st.button("📄 View Full Report", key=f"view_{result['id']}"):
                        st.info("Full report viewing feature - to be implemented")
                with col_b:
                    if st.button("📧 Share with Doctor", key=f"share_{result['id']}"):
                        st.info("Share functionality - to be implemented")
    
    def show_manage_images(self):
        """Main manage images page."""
        st.subheader("Manage My Images")
        st.info("Delete or reassign images that haven't been analyzed yet. Images with completed results cannot be deleted.")
        
        # Load data
        images_result = self.api_client.get_my_images()
        results_result = self.api_client.get_my_results()
        
        if not images_result["success"] or not results_result["success"]:
            st.error("❌ Failed to load image data")
            return
        
        images = images_result["data"]
        results = results_result["data"]
        
        # Show management interface
        self._show_manage_images(images, results)
    
    def _show_uploaded_images_management(self, images: list[Dict], results: list[Dict]):
        """Show uploaded images with management options."""
        st.markdown("#### Manage Your Uploaded Images")
        
        if not images:
            st.info("No images uploaded yet. Use the Upload tab to add medical images.")
            return
        
        # Load neurologists for reassignment
        neurologists_result = self.api_client.get_neurologists()
        neurologists = neurologists_result["data"] if neurologists_result["success"] else []
        
        for image in images:
            # Check if this image has completed results
            has_results = any(r['image_id'] == image['id'] for r in results)
            
            # Create status indicator
            status = "✅ Completed" if has_results else "⏳ Pending Analysis"
            status_color = "green" if has_results else "orange"
            
            with st.expander(f"📁 {image['study_name'] or 'Unnamed Study'} - {status}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**File:** {image['original_filename']}")
                    st.write(f"**Study Name:** {image['study_name'] or 'Not specified'}")
                    st.write(f"**Uploaded:** {image['uploaded_at'][:16]}")
                    st.markdown(f"**Status:** :{status_color}[{status}]")
                
                with col2:
                    if has_results:
                        # If completed, show view results button
                        if st.button("📊 View Results", key=f"view_res_{image['id']}", use_container_width=True):
                            st.session_state.current_page = "patient_results"
                            st.rerun()
    
    def _show_completed_studies(self, results: list[Dict]):
        """Show completed studies with link to full results."""
        st.markdown("#### Completed Medical Studies")
        
        if not results:
            st.info("No completed studies yet. Results will appear here once neurologists complete their analysis.")
            return
        
        for result in results:
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**📋 {result['study_name'] or 'Medical Study'}**")
                    st.caption(f"File: {result['image_filename']}")
                
                with col2:
                    st.write(f"👨‍⚕️ Dr. {result['neurologist_name']}")
                    st.caption(f"Completed: {result['created_at'][:10]}")
                
                with col3:
                    if st.button("📊 View", key=f"view_completed_{result['id']}", use_container_width=True):
                        st.session_state.current_page = "patient_results"
                        st.rerun()
                
                st.divider()

    def _show_manage_images(self, images, results):
        """Show only deletable/reassignable images (no results)."""
        st.markdown("#### Delete or Reassign Your Images")
        
        # Only show images WITHOUT results (secure filtering)
        manageable_images = [img for img in images if not any(r['image_id'] == img['id'] for r in results)]
        
        if not manageable_images:
            st.info("No images available for deletion or reassignment. Images with completed results cannot be modified.")
            return
        
        # Load neurologists for reassignment
        neurologists_result = self.api_client.get_neurologists()
        neurologists = neurologists_result["data"] if neurologists_result["success"] else []
        
        for image in manageable_images:
            with st.expander(f"📁 {image['study_name'] or 'Unnamed Study'} - ⏳ Pending"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**File:** {image['original_filename']}")
                    st.write(f"**Study Name:** {image['study_name'] or 'Not specified'}")
                    st.write(f"**Uploaded:** {image['uploaded_at'][:16]}")
                    st.write("**Status:** ⏳ Pending Analysis")
                    st.info("💡 This image can be modified because it has no completed results")
                
                with col2:
                    st.markdown("**Management Options**")
                    
                    # Reassign neurologist (only for images without results)
                    if neurologists:
                        neurologist_options = [f"Dr. {n['first_name']} {n['last_name']}" for n in neurologists]
                        new_neurologist_idx = st.selectbox(
                            "Reassign to:",
                            range(len(neurologist_options)),
                            format_func=lambda x: neurologist_options[x],
                            key=f"neuro_select_{image['id']}"
                        )
                        
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if st.button("🔄 Reassign", key=f"reassign_{image['id']}", use_container_width=True):
                                # Double-check no results exist before reassigning (security)
                                has_results = any(r['image_id'] == image['id'] for r in results)
                                if has_results:
                                    st.error("❌ Cannot reassign: Results already exist for this image")
                                else:
                                    new_neurologist = neurologists[new_neurologist_idx]
                                    result = self.api_client.update_image_assignment(image['id'], new_neurologist['id'])
                                    if result["success"]:
                                        st.success("✅ Reassigned successfully!")
                                        st.rerun()
                                    else:
                                        st.error(f"❌ Reassignment failed: {result.get('error', 'Unknown error')}")
                        
                        with col_b:
                            # Delete with confirmation (only for images without results)
                            delete_key = f"confirm_delete_{image['id']}"
                            
                            if st.button("🗑️ Delete", key=f"delete_{image['id']}", use_container_width=True):
                                if st.session_state.get(delete_key, False):
                                    # Double-check no results exist before deleting (security)
                                    has_results = any(r['image_id'] == image['id'] for r in results)
                                    if has_results:
                                        st.error("❌ Cannot delete: Results already exist for this image")
                                        # Clear confirmation state
                                        if delete_key in st.session_state:
                                            del st.session_state[delete_key]
                                    else:
                                        # Second click - actually delete
                                        with st.spinner("Deleting image..."):
                                            result = self.api_client.delete_image(image['id'])
                                            if result["success"]:
                                                st.success("✅ Image deleted successfully!")
                                                # Clear confirmation state
                                                if delete_key in st.session_state:
                                                    del st.session_state[delete_key]
                                                st.rerun()
                                            else:
                                                st.error(f"❌ Delete failed: {result.get('error', 'Unknown error')}")
                                                # Clear confirmation state on failure too
                                                if delete_key in st.session_state:
                                                    del st.session_state[delete_key]
                                else:
                                    # First click - set confirmation
                                    st.session_state[delete_key] = True
                                    st.rerun()
                            
                            # Show confirmation message if waiting for confirmation
                            if st.session_state.get(delete_key, False):
                                st.warning("⚠️ Click Delete again to confirm")
