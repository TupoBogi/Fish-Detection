from Fish_Det_Vid import process_video_with_yolo
from Fish_Det_Img import process_image_with_yolo
from Fish_Det_Live import process_live_stream

if __name__ == "__main__":
    print("Выберите режим работы:")
    print("1 — Обработка видео")
    print("2 — Обработка изображения")
    print("3 — Обработка в реальном времени")
    mode = input("Ваш выбор (1/2/3): ").strip()

    if mode == "1":
        input_path = input("Введите путь до видеофайла: ").strip()
        output_path = input("Введите путь для сохранения результата: ").strip()

        process_video_with_yolo(
            model_path='Det_Fish.pt',
            input_video_path=input_path,
            output_video_path=output_path
        )

    elif mode == "2":
        input_path = input("Введите путь до изображения: ").strip()
        output_path = input("Введите путь для сохранения результата: ").strip()

        process_image_with_yolo(
            model_path='Det_Fish.pt',
            input_image_path=input_path,
            output_image_path=output_path
        )

    elif mode == "3":
        print("Запуск обработки потока с камеры")
        process_live_stream(model_path='Det_Fish.pt')

    else:
        print("Неверный выбор. Введите 1, 2 или 3.")
