# Import

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import os
from PIL import Image, ImageDraw, ImageFont, ImageOps
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import textwrap
import re

#####################################################################################################################################

# Funtions

def create_pdf(images):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    for img in images:
        img_width, img_height = img.size
        aspect = img_height / float(img_width)
        
        if img_width > width:
            img_width = width
            img_height = width * aspect
        
        img_reader = ImageReader(img)
        c.drawImage(img_reader, 0, height - img_height, width=img_width, height=img_height)
        c.showPage()

    c.save()
    buffer.seek(0)
    return buffer

def image_processing_page():
    st.title("Image Processing and PDF Generation")

    if 'filtered_df' not in st.session_state or st.session_state['filtered_df'] is None:
        st.warning("Please filter data in the DBD Data Filter page first.")
        return

    filtered_df = st.session_state['filtered_df']

    # Text positioning options
    text_position = st.selectbox("Choose text position", 
                                 ['top-left', 'top-right', 'bottom-left', 'bottom-right', 'center'])
    
    font_size = st.slider("Font size", min_value=10, max_value=50, value=20)
    
    text_color = st.color_picker("Text color", "#096CB3")
    
    bg_color = st.color_picker("Background color", "#000000")
    bg_opacity = st.slider("Background opacity", min_value=0, max_value=255, value=128)

    # Fixed image path
    image_path = "./picture.jpg"

    if os.path.exists(image_path):
        # Process images
        processed_images = []
        for _, row in filtered_df.iterrows():
            img = Image.open(image_path)
            img_with_text = add_text_to_image(img, row['ที่อยู่จัดส่ง'], 
                                              position=text_position,
                                              font_size=font_size,
                                              text_color=tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                                              bg_color=tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (bg_opacity,))
            processed_images.append(img_with_text)

        # Generate PDF
        pdf_buffer = create_pdf(processed_images)

        # Offer PDF for download
        st.download_button(
            label="Download PDF",
            data=pdf_buffer,
            file_name="processed_images.pdf",
            mime="application/pdf"
        )
    else:
        st.error(f"Image file not found at {image_path}")

def add_grid_and_ruler(image, grid_spacing=50, ruler_width=20):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Draw grid
    for x in range(0, width, grid_spacing):
        draw.line([(x, 0), (x, height)], fill=(200, 200, 200, 128), width=1)
    for y in range(0, height, grid_spacing):
        draw.line([(0, y), (width, y)], fill=(200, 200, 200, 128), width=1)
    
    # Draw rulers
    font = ImageFont.load_default()
    for x in range(0, width, grid_spacing):
        draw.text((x, 0), str(x), fill=(255, 0, 0), font=font, fontsize=15)
    for y in range(0, height, grid_spacing):
        draw.text((0, y), str(y), fill=(255, 0, 0), font=font, fontsize=15)
    
    return image

def add_text_to_image(image, company_name, address, 
                      company_font_size=20, address_font_size=20,
                      company_x=50, company_y=50,
                      address_x=50, address_y=100,
                      line_spacing=1.2, postal_char_spacing=0, postal_x=50, postal_y=150,
                      text_color=(9, 108, 179),  # Default color set to #096CB3
                      bg_color=(255, 255, 255, 0), rotation=0,
                      show_grid=False):
    
    draw = ImageDraw.Draw(image)
    width, height = image.size
    font_path = "./Kanit-Regular.ttf"  # Ensure this Thai font file is in the same directory
    try:
        company_font = ImageFont.truetype(font_path, company_font_size)
        address_font = ImageFont.truetype(font_path, address_font_size)
    except IOError:
        st.error(f"Font file not found at {font_path}. Using default font.")
        company_font = address_font = ImageFont.load_default()
    
    # Add grid and ruler only if show_grid is True
    if show_grid:
        image = add_grid_and_ruler(image)

    # Draw company name
    company_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    company_draw = ImageDraw.Draw(company_img)
    company_draw.text((company_x, height - company_y), company_name, font=company_font, fill=text_color)
    rotated_company = company_img.rotate(rotation, expand=0)
    image.paste(rotated_company, (0, 0), rotated_company)

    # Split address at the first comma
    first_line, rest_of_address = address.split(',', 1) if ',' in address else (address, '')
    
    # Draw address
    address_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    address_draw = ImageDraw.Draw(address_img)
    
    # Draw first line
    y_text = height - address_y
    address_draw.text((address_x, y_text), first_line.strip(), font=address_font, fill=text_color)
    
    # Draw the rest of the address (if any)
    if rest_of_address:
        y_text += (address_font.getbbox(first_line)[3] - address_font.getbbox(first_line)[1]) + (line_spacing - 1) * address_font_size
        address_draw.text((address_x, y_text), rest_of_address.strip(), font=address_font, fill=text_color)

    # Draw postal code (assuming it's the last 5 digits of the address)
    postal_code = ''.join(filter(str.isdigit, address))[-5:]
    if postal_code:
        y_text += (address_font.getbbox(rest_of_address)[3] - address_font.getbbox(rest_of_address)[1]) + (line_spacing - 1) * address_font_size
        for i, char in enumerate(postal_code):
            char_width = address_font.getbbox(char)[2] - address_font.getbbox(char)[0]
            address_draw.text((postal_x + i * (char_width + postal_char_spacing), height - postal_y), 
                              char, font=address_font, fill=text_color)
    # # Draw address
    # address_img = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    # address_draw = ImageDraw.Draw(address_img)
    
    # lines = address.split(',')
    # y_text = height - address_y
    # for line in lines[:-1]:  # All lines except the last one (postal code)
    #     address_draw.text((address_x, y_text), line.strip(), font=address_font, fill=text_color)
    #     y_text += (address_font.getbbox(line)[3] - address_font.getbbox(line)[1]) + (line_spacing - 1) * address_font_size
    
    # # Draw postal code
    # postal_code = lines[-1].strip()
    # postal_match = re.search(r'(\d{5})(?:\.0)?', postal_code)
    # if postal_match:
    #     postal_code = postal_match.group(1)
    # else:
    #     postal_code = ''.join(filter(str.isdigit, postal_code))
    
    for i, char in enumerate(postal_code):
        char_width = address_font.getbbox(char)[2] - address_font.getbbox(char)[0]
        address_draw.text((postal_x + i * (char_width + postal_char_spacing), height - postal_y), 
                          char, font=address_font, fill=text_color)
      
    rotated_address = address_img.rotate(rotation, expand=0)
    image.paste(rotated_address, (0, 0), rotated_address)
    
    return image

def get_position(position, img_width, img_height, text_width, text_height, padding=10):
    if position == 'top-left':
        return (padding, padding)
    elif position == 'top-right':
        return (img_width - text_width - padding, padding)
    elif position == 'bottom-left':
        return (padding, img_height - text_height - padding)
    elif position == 'bottom-right':
        return (img_width - text_width - padding, img_height - text_height - padding)
    elif position == 'center':
        return ((img_width - text_width) // 2, (img_height - text_height) // 2)
    else:
        raise ValueError("Invalid position")

def load_multiple_files(uploaded_files):
    dfs = []
    for uploaded_file in uploaded_files:
        df = load_data(uploaded_file)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try Thai Windows encoding
        df = pd.read_csv(uploaded_file, encoding='cp874')
    
    # Convert 'ทุนจดทะเบียน' to numeric, handling comma separators
    df['ทุนจดทะเบียน'] = df['ทุนจดทะเบียน'].replace({',':''}, regex=True).astype(float)
    
    # Create the 2-digit purpose code column if it doesn't exist
    if 'หมวดย่อย_new' not in df.columns:
        if 'รหัสวัตถุประสงค์' in df.columns:
            df['หมวดย่อย_new'] = df['รหัสวัตถุประสงค์'].astype(str).str[:2]
        else:
            st.error(f"Column 'รหัสวัตถุประสงค์' not found in file {uploaded_file.name}. Please check your CSV file.")
    
    return df

def load_dbd_library():
    try:
        # Attempt to read the CSV file from the same directory as the script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(script_dir, 'dbd_id_library.csv')
        
        # Read the CSV file with proper encoding and handling of newlines within cells
        df = pd.read_csv(file_path, encoding='utf-8-sig', quotechar='"', escapechar='\\')
        
        # Clean up the data
        df['หมวดย่อย'] = df['หมวดย่อย'].astype(str).str.zfill(2)  # Ensure 2-digit codes
        df['ชื่อ'] = df['ชื่อ'].str.replace('\n', ' ').str.strip()  # Remove newlines and extra spaces
        
        return df
    except Exception as e:
        st.error(f"Error loading dbd_id_library.csv: {str(e)}")
        return pd.DataFrame(columns=['หมวดย่อย', 'ชื่อ'])

def filter_data(df, provinces, subcategories, min_capital, max_capital, exclude_provinces, exclude_subcategories):
    if provinces:
        df = df[df['จังหวัด'].isin(provinces) if not exclude_provinces else ~df['จังหวัด'].isin(provinces)]
    if subcategories:
        df = df[df['หมวดย่อย_new'].isin(subcategories) if not exclude_subcategories else ~df['หมวดย่อย_new'].isin(subcategories)]
    df = df[(df['ทุนจดทะเบียน'] >= min_capital) & (df['ทุนจดทะเบียน'] <= max_capital)]
    return df

def plot_capital_distribution(df):
    fig = go.Figure()
    fig.add_trace(go.Violin(
        y=df['ทุนจดทะเบียน'],
        box_visible=True,
        line_color='blue',
        meanline_visible=True,
        fillcolor='lightblue',
        opacity=0.6,
        x0='Registered Capital'
    ))

    fig.update_layout(
        title='Distribution of Registered Capital',
        xaxis_title='',
        yaxis_title='Registered Capital',
        xaxis=dict(rangeslider=dict(visible=True)),
    )
    st.plotly_chart(fig)

def plot_top_subcategories(df, top_n):
    subcategory_counts = df['หมวดย่อย_new'].value_counts().nlargest(top_n)
    fig = px.pie(values=subcategory_counts.values, names=subcategory_counts.index, 
                 title=f'Top {top_n} Subcategories')
    st.plotly_chart(fig)

def combine_address_columns(df):
    def format_address(row):
        # Clean up the postal code
        postal_code = str(row['รหัสไปรษณีย์'])
        postal_code = re.sub(r'\.0$', '', postal_code)  # Remove trailing .0 if present

        if row['จังหวัด'] == 'กรุงเทพมหานคร':
            return f"{row['ที่ตั้งสำนักงานใหญ่']}, {row['ตำบล']}, {row['อำเภอ']}, {row['จังหวัด']}, {postal_code}"
        else:
            return f"{row['ที่ตั้งสำนักงานใหญ่']}, ตำบล{row['ตำบล']}, อำเภอ{row['อำเภอ']}, จังหวัด{row['จังหวัด']}, {postal_code}"

    df['ที่อยู่จัดส่ง'] = df.apply(format_address, axis=1)
    return df
#####################################################################################################################################
# Main Interface

def main():
    st.set_page_config(page_title="DBD Data Filter", layout="wide", initial_sidebar_state="expanded")
    
    # Add a sidebar for page selection
    page = st.sidebar.selectbox("Choose a page", ["DBD Data Filter", "Image Processing"])
    
    if page == "DBD Data Filter":
        dbd_data_filter_page()
    elif page == "Image Processing":
        image_processing_page()

def dbd_data_filter_page():    
    st.title('DBD Data Filter and Export')
    
    dbd_library = load_dbd_library()
    
    uploaded_files = st.file_uploader("Choose CSV files", type="csv", accept_multiple_files=True)
    
    if uploaded_files:
        try:
            df = load_multiple_files(uploaded_files)
            df = combine_address_columns(df)
            st.write(f"Data loaded successfully. Total records: {len(df):,}")
        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            return

        if df.empty:
            st.warning("The loaded data is empty. Please check your CSV files.")
            return

        st.sidebar.title('Filter Options')

        with st.sidebar.expander("Reminder Note", expanded=False):
            st.write("""
            วิธีในการ Filter เบื้องต้น:
            - หมวดหมู่ : '47', '55', '56', '63', '68'
            - จังหวัด : กรุงเทพมหานคร
            - ทุนจดทะเบียน : 500k - 5M
            """)
        
        st.sidebar.subheader('Province Filter')
        province_options = sorted(df['จังหวัด'].dropna().unique())
        provinces = st.sidebar.multiselect('Select Provinces', options=province_options)
        exclude_provinces = st.sidebar.checkbox('Exclude Provinces')
                    
        st.sidebar.subheader('Subcategory Filter')
        subcategory_options = sorted(df['หมวดย่อย_new'].unique())
        subcategory_mapping = dict(zip(dbd_library['หมวดย่อย'], dbd_library['ชื่อ']))
        subcategory_options_with_names = [f"{code} - {subcategory_mapping.get(code, 'Unknown')}" for code in subcategory_options]
        selected_subcategories = st.sidebar.multiselect('Select Categories', options=subcategory_options_with_names)
        subcategories = [s.split(' - ')[0] for s in selected_subcategories]
        exclude_subcategories = st.sidebar.checkbox('Exclude Categories')

        st.sidebar.subheader("Capital Range")
        min_capital = df['ทุนจดทะเบียน'].min()
        max_capital = df['ทุนจดทะเบียน'].max()
        min_capital_input = st.sidebar.number_input("Min Capital", min_value=float(min_capital), max_value=float(max_capital), value=float(min_capital), format="%0.2f")
        max_capital_input = st.sidebar.number_input("Max Capital", min_value=float(min_capital), max_value=float(max_capital), value=float(max_capital), format="%0.2f")
        
        if st.sidebar.button('Apply Filters', key='apply_filters'):
            try:
                filtered_df = filter_data(df, provinces, subcategories, min_capital_input, max_capital_input, exclude_provinces, exclude_subcategories)
                st.session_state['filtered_df'] = filtered_df
                st.session_state['filter_applied'] = True
            except Exception as e:
                st.error(f"Error applying filters: {str(e)}")
                return
        
        if 'filter_applied' in st.session_state and st.session_state['filter_applied']:
            st.subheader('Filtered Results')
            filtered_df = st.session_state['filtered_df']
            total_companies = len(filtered_df)
            st.write(f"Total filtered companies: {total_companies:,}")
            
            if total_companies > 0:
                items_per_page = 20
                num_pages = (total_companies - 1) // items_per_page + 1
                page = st.number_input('Page', min_value=1, max_value=num_pages, value=1)
                start_idx = (page - 1) * items_per_page
                end_idx = start_idx + items_per_page
                
                display_df = filtered_df.iloc[start_idx:end_idx].copy()
                display_df.insert(0, 'ลำดับ', range(start_idx + 1, min(end_idx + 1, total_companies + 1)))
                
                display_df['หมวดย่อย_name'] = display_df['หมวดย่อย_new'].map(subcategory_mapping)
                
                cols = display_df.columns.tolist()
                cols.insert(cols.index('หมวดย่อย_new') + 1, cols.pop(cols.index('หมวดย่อย_name')))
                display_df = display_df[cols]
                
                st.dataframe(display_df.reset_index(drop=True), height=400)
                
                columns_to_drop = ['ที่ตั้งสำนักงานใหญ่', 'ตำบล', 'อำเภอ', 'จังหวัด', 'รหัสไปรษณีย์']
                export_df = filtered_df.drop(columns=columns_to_drop)
                csv = export_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="Download filtered data as CSV",
                    data=csv,
                    file_name="filtered_data.csv",
                    mime="text/csv",
                )
            else:
                st.write("No companies match the selected filters.")
    else:
        st.write("Please upload CSV files to start filtering data.")
#####################################################################################################################################
def image_processing_page():
    st.title("Image Processing and PDF Generation")

    if 'filtered_df' not in st.session_state or st.session_state['filtered_df'] is None:
        st.warning("Please filter data in the DBD Data Filter page first.")
        return

    filtered_df = st.session_state['filtered_df']

    # Move all adjustments to sidebar
    st.sidebar.title("Adjustment Controls")

    st.sidebar.subheader("Text Formatting")
    font_size = st.sidebar.number_input("Font size", min_value=10, max_value=50, value=20)
    line_spacing = st.sidebar.number_input("Line spacing", min_value=1.0, max_value=2.0, value=1.85, step=0.05)
    text_color = st.sidebar.color_picker("Text color", "#096CB3")  # Default to black
    rotation = st.sidebar.number_input("Rotation (degrees)", min_value=0, max_value=360, value=0)

    st.sidebar.subheader("Company Name Position")
    company_x = st.sidebar.number_input("Company X (from left)", min_value=0, max_value=1000, value=230)
    company_y = st.sidebar.number_input("Company Y (from bottom)", min_value=0, max_value=1000, value=490)

    st.sidebar.subheader("Address Position")
    address_x = st.sidebar.number_input("Address X (from left)", min_value=0, max_value=1000, value=230)
    address_y = st.sidebar.number_input("Address Y (from bottom)", min_value=0, max_value=1000, value=435)

    st.sidebar.subheader("Postal Code")
    postal_char_spacing = st.sidebar.number_input("Postal code character spacing", min_value=0, max_value=100, value=38)
    postal_x = st.sidebar.number_input("Postal code X (from left)", min_value=0, max_value=1000, value=373)
    postal_y = st.sidebar.number_input("Postal code Y (from bottom)", min_value=0, max_value=1000, value=310)

    st.sidebar.subheader("Background")
    bg_color = st.sidebar.color_picker("Background color", "#FFFFFF")  # Default to white
    bg_opacity = st.sidebar.number_input("Background opacity", min_value=0, max_value=255, value=0)  # Default to transparent

    # Main content area
    if len(filtered_df) > 0:
        preview_company = st.text_input("Edit preview company name", filtered_df.iloc[0]['ชื่อนิติบุคคล'])
        preview_address = st.text_area("Edit preview address (commas will create new lines)", filtered_df.iloc[0]['ที่อยู่จัดส่ง'])

    if st.button("Update Preview"):
        # Fixed image path
        image_path = "/Users/Workspace/CODE-WorkingSpace/Station-DBD-Filter/WebApp/picture.jpg"

        if os.path.exists(image_path):
            img = Image.open(image_path)
            preview_img = add_text_to_image(img.copy(), preview_company, preview_address,
                                            company_font_size=font_size,
                                            address_font_size=font_size,
                                            company_x=company_x,
                                            company_y=company_y,
                                            address_x=address_x,
                                            address_y=address_y,
                                            line_spacing=line_spacing,
                                            postal_char_spacing=postal_char_spacing,
                                            postal_x=postal_x,
                                            postal_y=postal_y,
                                            text_color=tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                                            bg_color=tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (bg_opacity,),
                                            rotation=rotation,
                                            show_grid=True)  # Show grid for preview
            
            st.image(preview_img, caption="Preview", use_column_width=True)
        else:
            st.error(f"Image file not found at {image_path}")


    if st.button("Generate PDF"):
        # Fixed image path
        image_path = "/Users/Workspace/CODE-WorkingSpace/Station-DBD-Filter/WebApp/picture.jpg"

        if os.path.exists(image_path):
            img = Image.open(image_path)
            processed_images = []
            
            # Create a progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_rows = len(filtered_df)

            for index, row in filtered_df.iterrows():
                img_with_text = add_text_to_image(img.copy(), row['ชื่อนิติบุคคล'], row['ที่อยู่จัดส่ง'],
                                                company_font_size=font_size,
                                                address_font_size=font_size,
                                                company_x=company_x,
                                                company_y=company_y,
                                                address_x=address_x,
                                                address_y=address_y,
                                                line_spacing=line_spacing,
                                                postal_char_spacing=postal_char_spacing,
                                                postal_x=postal_x,
                                                postal_y=postal_y,
                                                text_color=tuple(int(text_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)),
                                                bg_color=tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + (bg_opacity,),
                                                rotation=rotation,
                                                show_grid=False)  # Don't show grid for PDF
                processed_images.append(img_with_text)
                 
                # Update progress bar
                progress = min(1.0, (index + 1) / total_rows)
                progress_bar.progress(progress)
                status_text.text(f"Processing image {index + 1} of {total_rows}")

            # Generate PDF
            status_text.text("Generating PDF...")
            pdf_buffer = create_pdf(processed_images)
            
            # Complete the progress bar
            progress_bar.progress(1.0)
            status_text.text("PDF generation complete!")

            # Offer PDF for download
            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name="processed_images.pdf",
                mime="application/pdf"
            )
        else:
            st.error(f"Image file not found at {image_path}")
#####################################################################################################################################

if __name__ == "__main__":
    main()
