# ========== Home Page ========== #
if page == "Home":
    # ==== Styling ==== #
    st.markdown(
        """
        <style>
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            color: #4B8BBE;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 20px;
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ==== Header ==== #
    st.markdown('<div class="title">🐶🐱 Cat vs Dog Classifier</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload an image and let AI guess!</div>', unsafe_allow_html=True)
    st.markdown("---")

    # ==== Upload + Guess ==== #
    uploaded = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"])
    guess = st.radio("🤔 What do **you** think it is?", ["Not Sure", "Cat", "Dog"])

    # ==== If Image Uploaded ==== #
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="🖼️ Uploaded Image", use_container_width=True)

        with st.spinner("🧠 Analyzing..."):
            label, conf = predict_label(img)

        # Result Display
        emoji = "🐱" if label == "cat" else "🐶"
        st.success(f"{emoji} It's a **{label.upper()}** with {conf*100:.2f}% confidence!")

        # Confidence Plot
        plot_confidence(conf)

        # ==== If Correct Guess ==== #
        if guess.lower() == label.lower() and guess != "Not Sure":
            # Celebrate Correct Guess
            st.balloons()  # Add Balloons for Celebration 🎈
            # Play the correct animal sound based on the prediction
            audio_path = "Deployment/cat.mp3" if label == "cat" else "Deployment/dog.mp3"
            st.audio(audio_path, format="audio/mp3")
            st.success("🎉 Great job! You guessed it right! 🐱🐶")
            
            # Optional Lottie Animation (if available)
            animal_party_url = "https://lottie.host/7b92b97a-9aeb-42df-bd91-622d8eb80347/hGMpbibYrh.json"
            try:
                animal_party_json = load_lottie_url(animal_party_url)
                if animal_party_json:
                    st_lottie(animal_party_json, height=300, key="celebrate")
                else:
                    st.write("🎉 Lottie animation failed to load.")
            except Exception as e:
                st.write(f"🎉 Lottie animation error: {str(e)}")
                st.write("🎉 Let's celebrate anyway!")

        # ==== If Incorrect Guess ==== #
        elif guess != "Not Sure":
            st.warning("😿 Oops! Try again, you're close!")

        # Cute Animal Sound 🐾
        audio_path = "Deployment/cat.mp3" if label == "cat" else "Deployment/dog.mp3"
        st.audio(audio_path, format="audio/mp3", start_time=0)

        # Optional Voice Output
        if st.toggle("🔈 Hear it"):
            speak(f"It's a {label} with {conf*100:.2f} percent confidence.")
