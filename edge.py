#!/usr/bin/python3

import cv2
import numpy as np

# トラックバーのコールバック関数（何もしない）
def nothing(x):
    pass

# ウィンドウ作成とトラックバー(GUIスライダー)追加
cv2.namedWindow('Canny Edge')

cv2.createTrackbar("Threshold1", "Canny Edge", 50, 500, nothing)
cv2.createTrackbar("Threshold2", "Canny Edge", 150, 500, nothing)

def canny_edge_demo():
    # カメラ起動（0はデフォルトカメラ）
    # 2がウェブカメラ
    #カメラから映像を読み込む
    #解像度を640×480に設定
    cap = cv2.VideoCapture(2)
    cap.set(3, 640) #幅
    cap.set(4, 480) #高さ

    #映像フレームごとに処理
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # トラックバーからしきい値を取得(thはトラックバー)
        th1 = cv2.getTrackbarPos("Threshold1", "Canny Edge")
        th2 = cv2.getTrackbarPos("Threshold2", "Canny Edge")

        # グレースケール化(これで白黒にしてる)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #明るさ補正
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        #ノイズ除去　ガウシアン
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Cannyエッジ検出(輪郭を検出)
        edges = cv2.Canny(blurred, th1, th2)

        #モルフォロジー
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=2)

        # 表示
        cv2.imshow("Original", frame)
        cv2.imshow("Canny Edge", edges)
        cv2.imshow("Morph Gradient", gradient)

        # ESCキーで終了
        k = cv2.waitKey(10)
        if k == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    canny_edge_demo()
