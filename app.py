# 3.10 버전이하에서만 작동합니다.
# conda create -n test2 python=3.10
from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import plotly.express as px
import numpy as np
import streamlit as st
import datetime 

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r', encoding="utf-8").readlines()

st.set_page_config(layout="wide")
st.title("🍊감귤류 분류하기")
# 선택 옵션: 카메라 입력 또는 파일 업로드
input_method = st.selectbox("이미지 입력 방식", ["카메라 사용", "파일 업로드"], help=None)

# ✅ 메인 화면을 이분할
col1, col2 = st.columns([1, 1]) 

with col1 : 
    if input_method == "카메라 사용":
        img_file_buffer = st.camera_input("정중앙에 사물을 위치하고 사진찍기 버튼을 누르세요")
    else:
        img_file_buffer = st.file_uploader("이미지 파일 업로드", type=["png", "jpg", "jpeg"])

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)  # 이미지 불러오기
        st.image(image, caption="업로드한 이미지", use_container_width=True)
# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
# 들어온 이미지를 224 x 224 x 3차원으로 변환하기 위해서 빈 벡터를 만들어 놓음
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# 이전 예측 결과 저장
if "history" not in st.session_state:
    st.session_state["history"] = []

if img_file_buffer is not None:
    # 이미지명 추출 
    image_name = img_file_buffer.name if hasattr(img_file_buffer, "name") else "카메라 이미지"

    # # To read image file buffer as a PIL Image:
    # image = Image.open(img_file_buffer) # 입력받은 사진을 행렬로 변환

    # # To convert PIL Image to numpy array:
    # img_array = np.array(image) # ndarray로 변환

    # Replace this with the path to your image
    # 원본 이미지 불러오기
     
    image = Image.open(img_file_buffer).convert('RGB')

    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    # 모델에 들어갈 수 있는 224 x 224 사이즈로 변환 
    # 보간 방식 : Image.Resampling.LANCZOS 
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    #turn the image into a numpy array
    # 이미지를 넘파이 행렬로 변환 
    image_array = np.asarray(image)

    # Normalize the image
    # 모델이 학습했을 때 Nomalize 한 방식대로 이미지를 Nomalize 
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Load the image into the array
    # 빈 ARRAY에 전처리를 완료한 이미지를 복사
    data[0] = normalized_image_array

    # run the inference
    # h5 모델에 예측 의뢰 
    prediction = model.predict(data)
    

    # 높은 신뢰도가 나온 인덱의 인덱스 자리를 저장
    index = np.argmax(prediction)

    # labels.txt 파일에서 가져온 값을 index로 호출
    # 좋아하는 만화 선택하세요 - 만화 제목(text 리스트)랑 img 경로 리스트 일치 시킬 때 인덱스 활용한 것과 같은 방법
    class_name = class_names[index]
    class_probs = {class_names[i][2:]: round(prediction[0][i] * 100, 2) for i in range(len(class_names))}

    # 예측 결과에서 신뢰도를 꺼내 옵니다  
    confidence_score = prediction[0][index]
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with col2 : 
        st.header("")
        st.header("")
        st.header("출력 결과")
        st.markdown(f'##### Class: {class_name[2:]}')
        st.markdown(f'##### Confidence score: {confidence_score:.2f}%')
        
        # ✅ Plotly 막대 그래프 생성
        fig = px.bar(
            x=list(class_probs.values()), 
            y=list(class_probs.keys()), 
            orientation='h',  # 가로 막대 그래프
            text=[f"{v}%" for v in class_probs.values()],  # 각 막대에 확률 표시
            labels={"x": "Confidence (%)", "y": "Classes"},
            color=list(class_probs.keys()),  # 클래스별 색상 지정
            color_discrete_sequence=px.colors.qualitative.Pastel  # ✅ 색상 테마 변경
        )

        # ✅ 그래프 커스텀 스타일 적용
        fig.update_traces(
            textposition="inside",  # 막대 내부에 확률값 표시
            marker=dict(line=dict(width=1.5, color='black'))  # ✅ 막대 테두리 추가
        )
        fig.update_layout(
            title="🍊 감귤류 분류 예측 확률",
            title_font=dict(size=20, color="black"),  # ✅ 제목 스타일 추가
            xaxis=dict(title="Confidence Score (%)", showgrid=True, zeroline=True),  # ✅ X축 조정
            yaxis=dict(title="", showgrid=False),  # ✅ Y축 제목 숨김
            height=350,  # ✅ 그래프 크기 조정
            margin=dict(l=50, r=50, t=50, b=50),  # ✅ 마진 조정 (여백 최적화)
            paper_bgcolor="white",  # ✅ 배경색 변경
            plot_bgcolor="white"  # ✅ 그래프 배경 변경
        )

        # ✅ Streamlit에서 그래프 표시
        st.plotly_chart(fig, use_container_width=True)

    st.session_state["history"].append({
        "image_name": image_name,
        "class": class_name[2:],  # 앞에 공백 제거
        "confidence": round(confidence_score, 4),
        "time": timestamp
    })

# 🔹 사이드바에 저장된 예측 결과 표시
st.sidebar.header("📜 예측 기록")
for item in reversed(st.session_state["history"]):  # 최신 항목이 위로 오도록 reverse
    st.sidebar.write(f"🕒 {item['time']}")
    st.sidebar.write(f"📁 **이미지:** {item['image_name']}")
    st.sidebar.write(f"🔍 **Class:** {item['class']}")
    st.sidebar.write(f"🎯 **Confidence:** {item['confidence']:.4f}")
    st.sidebar.write("---")  # 구분선 추가