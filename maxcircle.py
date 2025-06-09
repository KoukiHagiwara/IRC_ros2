#!/usr/bin/python3

#一番手前にある手前の円を検出
#最大の円だけ描画

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
        
        # 赤色の低い方の範囲（0〜10）
        lower_red1 = np.array([0, 58, 30])
        upper_red1 = np.array([10, 255, 255])

        # 赤色の高い方の範囲（170〜180）
        lower_red2 = np.array([170, 58, 30])
        upper_red2 = np.array([180, 255, 255])
        
        draw_color = {
            "red": (0,0,255),
            "blue": (255,0,0),
            "yellow": (0,255,255)
        }

        # 2つのマスクを作成して合成
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask_red = cv2.bitwise_or(mask1, mask2)

        # 青
        lower_blue = np.array([96, 100, 0])
        upper_blue = np.array([159, 250, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        #黄
        lower_yellow = np.array([22, 105, 81])
        upper_yellow = np.array([69, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 全色を統合
        mask = cv2.bitwise_or(mask_red, mask_blue)
        mask = cv2.bitwise_or(mask, mask_yellow)
        # 元画像とマスクを合成（該当色だけ残す）
        result = cv2.bitwise_and(img, img, mask=mask)

        masks = {
            "red": mask_red,
            "blue": mask_blue,
            "yellow": mask_yellow
        }
        
        # 最大円を記録する変数
        max_radius = 0
        max_center = None
        max_color = None

        for color, mask in masks.items():
            color_bgr = draw_color[color]  # 色を取得
        
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

            if th1 <= 0: th1 = 50
            if th2 <= 0: th2 = 150

            edges = cv2.Canny(blurred, th1, th2)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(edges, cv2.MORPH_GRADIENT, kernel, iterations=2)

            circles = cv2.HoughCircles(
                blurred, cv2.HOUGH_GRADIENT,
                dp=1, minDist=30,
                param1=th2, param2=35,
                minRadius=25, maxRadius=130
            )

            if circles is not None:
                circles = np.uint16(np.around(circles))
                for (x, y, r) in circles[0]:
                    if r > max_radius:
                        max_radius = r
                        max_center = (x, y)
                        max_color = color
        # 最大の円を描画    
        if max_center is not None:
            cv2.circle(img, max_center, int(max_radius), draw_color[max_color], 2)
            cv2.circle(img, max_center, 5, (0, 0, 0), -1)
            cv2.putText(img, f"{max_color.capitalize()} Ball Pos: {max_center}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, draw_color[max_color], 2)



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
