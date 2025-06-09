#!/usr/bin/python3

import cv2
import numpy as np

# トラックバーのコールバック関数（何もしない）
def nothing(x):
    pass

# ウィンドウの作成（トラックバー用）
cv2.namedWindow('HSV')

# トラックバーを作成（各値はHSVの範囲に基づく）
cv2.createTrackbar("H_l", "HSV", 0, 180, nothing)     # 色相の下限
cv2.createTrackbar("H_h", "HSV", 180, 180, nothing)   # 色相の上限
cv2.createTrackbar("S_l", "HSV", 0, 255, nothing)     # 彩度の下限
cv2.createTrackbar("S_h", "HSV", 255, 255, nothing)   # 彩度の上限
cv2.createTrackbar("V_l", "HSV", 0, 255, nothing)     # 明度の下限
cv2.createTrackbar("V_h", "HSV", 255, 255, nothing)   # 明度の上限

cv2.createTrackbar("Threshold1", "HSV", 50, 500, nothing)
cv2.createTrackbar("Threshold2", "HSV", 150, 500, nothing)

def ball():
    # カメラを起動（0は通常PC内蔵カメラ）
    # ウェブカメラ 2
    cap = cv2.VideoCapture(2)
    cap.set(3, 640)  # 幅
    cap.set(4, 480)  # 高さ

    while True:
        ret, img = cap.read()  # フレームを取得
        if not ret:
            break
       #HSV 色彩　彩度　明度
        # トラックバーからHSVの各しきい値を取得
        h_l = cv2.getTrackbarPos("H_l", "HSV")
        h_h = cv2.getTrackbarPos("H_h", "HSV")
        s_l = cv2.getTrackbarPos("S_l", "HSV")
        s_h = cv2.getTrackbarPos("S_h", "HSV")
        v_l = cv2.getTrackbarPos("V_l", "HSV")
        v_h = cv2.getTrackbarPos("V_h", "HSV")
        
        # BGRからHSVに変換
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # マスクの範囲を指定
        lower = np.array([h_l, s_l, v_l])
        upper = np.array([h_h, s_h, v_h])

        # 指定範囲の色を抽出（白：該当、黒：非該当）
        mask = cv2.inRange(hsv, lower, upper)

        # 元画像とマスクを合成（該当色だけ残す）
        result = cv2.bitwise_and(img, img, mask=mask)

        # グレースケール化(これで白黒にしてる)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)

        #明るさ補正
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        #ノイズ除去　ガウシアン
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)


         # トラックバーからしきい値を取得(thはトラックバー)
        th1 = cv2.getTrackbarPos("Threshold1", "HSV")
        th2 = cv2.getTrackbarPos("Threshold2", "HSV")


        # Cannyエッジ検出(輪郭を検出)
        edges = cv2.Canny(blurred, th1, th2)

        #モルフォロジー
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=1)

        # 結果を表示
        cv2.imshow('Original', img)    # 元画像
        cv2.imshow('Mask', result)     # 色抽出結果
        cv2.imshow("Edge on Mask", gradient)

        # Escキー（ASCIIコード27）を押すと終了
        k = cv2.waitKey(10)
        if k == 27:
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

# メイン関数の呼び出し
if __name__ == "__main__":
    ball()
