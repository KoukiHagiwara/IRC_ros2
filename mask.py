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

def mask_ball():
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
       # lower = np.array([23, 80, 53])
       # upper = np.array([180, 255, 238])


        # 指定範囲の色を抽出（白：該当、黒：非該当）
        mask = cv2.inRange(hsv, lower, upper)

        # 元画像とマスクを合成（該当色だけ残す）
        result = cv2.bitwise_and(img, img, mask=mask)

        # 結果を表示
        cv2.imshow('Original', img)    # 元画像
        cv2.imshow('Mask', result)     # 色抽出結果

        # Escキー（ASCIIコード27）を押すと終了
        k = cv2.waitKey(10)
        if k == 27:
            break

    # 後処理
    cap.release()
    cv2.destroyAllWindows()

# メイン関数の呼び出し
if __name__ == "__main__":
    mask_ball()
