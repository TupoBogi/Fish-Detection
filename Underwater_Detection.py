from Fish_Detection import process_video_with_yolo

if __name__ == "__main__":
    input1 = input("Введите пути до исходного видео и для сохранения обработанного видео: ").strip()
    input2 = input().strip()

    process_video_with_yolo(
        model_path='Det_Fish.pt',
        input_video_path=input1,
        output_video_path=input2
    )
