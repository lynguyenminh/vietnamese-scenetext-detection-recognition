import streamlit as st

# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: 100% 100%;
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('5.webp')    



st.markdown('''
        # <center>Vietnamese Scene Text Detection and Recognition</center>
        <br>
        <br>
        <p style=" color:White; font-size: 42px;">New image</p>
        ''', unsafe_allow_html=True
    )


st.write('''

The Scene text problem for Vietnamese is a quite popular task in recent years. Its two main sub-problems are Text Detection used to locate the position of text in an image and Text Recognition used to read text in an image into text. 

Although not a new problem, it still presents many challenges for researchers. After a period of researching the problem, I have accumulated quite a lot of knowledge and experience to solve this problem. This is the API I have built to solve this problem.</p>

In this API, I use the following technologies: 
<br> - Text detection: YOLO, CRAFT
<br> - Text Recognition: Parseq, VietOCR
<br> - Streamlit to build the API.</p>
Besides the algorithms mentioned above, I will continue to research and update the best algorithms into this model.

<br>
<br>
<br>
<br>
Author: Nguyễn Minh Lý

Source code: https://github.com/lynguyenminh/scenetext-api
''', unsafe_allow_html=True)
