import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as MobileNetV2_preprocess_input

# Navigasi Sidebar
pilihan = st.sidebar.selectbox("Pilih Halaman", ["Deteksi Penyakit", "Artikel"])

if pilihan == "Deteksi Penyakit":
    # Judul Aplikasi
    st.title('Aplikasi Deteksi Dini Penyakit Daun Jagung')

    ## Info Aplikasi    
    st.markdown("""
    <h2>Tentang</h2>
    <p style='text-align: justify;'>
    Penyakit tanaman jagung merupakan ancaman serius bagi ketahanan pangan global, terutama bagi petani kecil yang sangat bergantung pada hasil panen mereka. Jagung, sebagai salah satu tanaman pangan utama, seringkali mengalami penurunan hasil panen yang signifikan akibat serangan penyakit. Di banyak negara berkembang, petani jagung dapat kehilangan lebih dari 50% hasil panen mereka karena penyakit.
    </p>
    <p style='text-align: justify;'>
    Dengan populasi dunia yang diperkirakan akan mencapai lebih dari 9,7 miliar pada tahun 2050, menjaga kesehatan tanaman jagung menjadi semakin penting. Oleh karena itu, identifikasi dini dan akurat terhadap penyakit daun jagung menjadi prioritas agar tindakan pencegahan dapat segera diambil.
    </p>
    <p style='text-align: justify;'>
    Aplikasi Streamlit ini memanfaatkan teknologi Pembelajaran Mendalam untuk mendeteksi berbagai penyakit pada daun jagung, termasuk Hawar Daun Utara, Karat Biasa, dan Bercak Daun Abu-abu, berdasarkan gambar digital.
    </p>
    """, unsafe_allow_html=True)


    ## Memuat Berkas
    st.sidebar.write("# Berkas Diperlukan")
    uploaded_image = st.sidebar.file_uploader('', type=['jpg','png','jpeg'])

    ################### Kelas dan Dataframe Probabilitas #############################
    # Pemetaan Kelas
    map_class = {
            0:'Blight',
            1:'Common Rust',
            2:'Grey Leaf Spot',
            3:'Healthy'
            }
            
    # Dataframe 
    dict_class = {
            'Kondisi Daun Jagung': ['Blight', 'Common Rust', 'Grey Leaf Spot', 'Healthy'],
            'Kepercayaan': [0,0,0,0]
            }
            
    df_results = pd.DataFrame(dict_class, columns = ['Kondisi Daun Jagung', 'Kepercayaan'])
        
    def predictions(preds):
        df_results.loc[df_results['Kondisi Daun Jagung'].index[0], 'Kepercayaan'] = preds[0][0]
        df_results.loc[df_results['Kondisi Daun Jagung'].index[1], 'Kepercayaan'] = preds[0][1]
        df_results.loc[df_results['Kondisi Daun Jagung'].index[2], 'Kepercayaan'] = preds[0][2]
        df_results.loc[df_results['Kondisi Daun Jagung'].index[3], 'Kepercayaan'] = preds[0][3]

        return (df_results)          

    ########################################### Memuat Model #########################
    #@st.cache
    def get_model():
        model = tf.keras.models.load_model("model_mobnetv2.h5")
        return model

    if __name__=='__main__':
        
        # Model
        model = get_model()

        # Praproses Gambar
        if not uploaded_image:
            st.sidebar.write('Silakan unggah gambar sebelum melanjutkan!')
            st.stop()
        else:
            # Dekode gambar dan prediksi kelas
            img_as_bytes = uploaded_image.read() # Encoding image
            st.write("## Gambar Daun Jagung")
            st.image(img_as_bytes, use_column_width= True) # Tampilkan gambar
            img = tf.io.decode_image(img_as_bytes, channels = 3) # Konversi gambar ke tensor
            img = tf.image.resize(img,(224,224)) # Ubah ukuran gambar
            img_arr = tf.keras.preprocessing.image.img_to_array(img) # Konversi gambar ke array
            img_arr = tf.expand_dims(img_arr, 0) # Buat batch

        img = MobileNetV2_preprocess_input(img_arr)

        Genrate_pred = st.button("Deteksi Hasil") 
    
        if Genrate_pred:
            st.subheader('Probabilitas berdasarkan Kelas') 
            preds = model.predict(img)
            preds_class = model.predict(img).argmax()

            st.dataframe(predictions(preds))

            if (map_class[preds_class]=="Blight") or (map_class[preds_class]=="Common Rust") or (map_class[preds_class]=="Grey Leaf Spot"): 
                st.subheader("Daun Jagung terinfeksi oleh Penyakit {}".format(map_class[preds_class]))

            else:
                st.subheader("Daun Jagung dalam kondisi {}".format(map_class[preds_class]))

elif pilihan == "Artikel":
    # Konten halaman Artikel
    st.title('Jenis-Jenis Penyakit Daun Jagung')

    # Penjelasan Jenis Penyakit, Pencegahan, dan Gambar dalam satu kolom memanjang ke bawah
    st.subheader('Blight')
    st.image('images/blight.jpg', caption='Blight', width=300)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Blight</strong> adalah penyakit yang disebabkan oleh berbagai patogen, tetapi dua yang paling umum pada jagung adalah Northern Corn Leaf Blight (NCLB) dan Southern Corn Leaf Blight (SCLB). Gejalanya biasanya daun jagung akan menunjukkan bercak-bercak besar yang berwarna abu-abu kehijauan dengan batas yang gelap. Bercak-bercak ini biasanya berbentuk panjang dan berwarna kecoklatan. Daun yang terinfeksi akan menguning dan akhirnya mati.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Pencegahan:</strong> Untuk mencegah Blight, lakukan rotasi tanaman secara teratur untuk mengurangi keberadaan patogen di dalam tanah. Pilih varietas jagung yang telah direkayasa untuk memiliki ketahanan genetik terhadap penyakit ini. Selain itu, pastikan penggunaan sistem irigasi yang efisien untuk menghindari kelembapan berlebih, dan terapkan sanitasi ketat dengan membersihkan peralatan pertanian serta membuang sisa tanaman yang terinfeksi.
        </div>
    """, unsafe_allow_html=True)

    st.subheader('Common Rust')
    st.image('images/common_rust.jpg', caption='Common Rust', width=300)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Common Rust</strong> disebabkan oleh jamur Puccinia sorghi. Gejala utamanya adalah munculnya bercak-bercak kecil berwarna merah atau oranye pada permukaan atas daun. Bercak ini bisa berkembang menjadi pustula yang penuh dengan spora. Daun yang terinfeksi akhirnya akan menguning dan kering.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Pencegahan:</strong> Pencegahan Common Rust memerlukan pemilihan varietas jagung yang tahan terhadap penyakit ini. Rotasi tanaman dan pengelolaan residu tanaman yang baik dapat mengurangi risiko infeksi. Pemantauan rutin dan penggunaan fungisida preventif pada tahap awal perkembangan penyakit sangat dianjurkan untuk mengendalikan penyebaran infeksi.
        </div>
    """, unsafe_allow_html=True)

    st.subheader('Grey Leaf Spot')
    st.image('images/grey_leaf_spot.jpg', caption='Grey Leaf Spot', width=300)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Grey Leaf Spot</strong> disebabkan oleh jamur Cercospora zeae-maydis. Gejala awalnya berupa bercak-bercak kecil yang berwarna abu-abu terang dengan batasan yang lebih gelap. Bercak ini dapat membesar dan bergabung, menyebabkan daun menguning dan mati secara bersamaan.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Pencegahan:</strong> Pencegahan Grey Leaf Spot melibatkan penggunaan varietas jagung yang tahan terhadap penyakit ini. Rotasi tanaman dan pengelolaan residu tanaman yang baik dapat mengurangi risiko infeksi. Penggunaan fungisida berbasis kuprum atau triazol pada fase awal penyakit juga efektif dalam menghambat penyebaran patogen.
        </div>
    """, unsafe_allow_html=True)

    st.subheader('Healthy')
    st.image('images/healthy.jpg', caption='Healthy', width=300)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Healthy</strong> Daun jagung yang sehat tidak menunjukkan tanda-tanda penyakit. Daun terlihat hijau segar tanpa bercak atau kerusakan yang terlihat. Menjaga kesehatan tanaman sangat penting untuk memastikan hasil panen yang optimal.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("""
        <div style='text-align: justify;'>
        <strong>Perawatan:</strong> Untuk menjaga kesehatan daun jagung, terapkan praktik agronomi yang baik seperti penggunaan pupuk yang seimbang dan pengelolaan air yang tepat. Pastikan juga jarak tanam yang cukup untuk mendukung sirkulasi udara yang baik, dan lakukan pemantauan rutin untuk deteksi dini terhadap potensi penyakit. Penggunaan metode IPM (Integrated Pest Management) yang efektif juga sangat penting untuk menjaga kesehatan tanaman secara keseluruhan.
        </div>
    """, unsafe_allow_html=True)


