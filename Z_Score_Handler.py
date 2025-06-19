import math


class Z_Score_Handler():
    def __init__(self, colors_for_image = None):
        super().__init__()
        self.colors_for_image = colors_for_image

    def __Z_score_algorithm(self):
        color_list = self.colors_for_image.get_colors_for_image()
        red_sum = 0
        green_sum = 0 
        blue_sum = 0
        for col in color_list:
            red_sum += col[0]
            green_sum += col[1]
            blue_sum += col[2]

        red_avg = red_sum / len(color_list)
        green_avg = green_sum / len(color_list)
        blue_avg = blue_sum / len(color_list)

        red_sum = 0
        green_sum = 0
        blue_sum = 0

        for col in color_list:
            red_sum += math.pow(col[0] - red_avg, 2)
            green_sum += math.pow(col[1] - green_avg, 2)
            blue_sum += math.pow(col[2] - blue_avg, 2)

        red_deviation = math.sqrt(red_sum / len(color_list))
        green_deviation = math.sqrt(green_sum / len(color_list))
        blue_deviation = math.sqrt(blue_sum / len(color_list))


        z_score_red = []
        z_score_green = []
        z_score_blue = []

        for col in color_list:
            z_score_red.append((col[0] - red_avg)/red_deviation)
            z_score_green.append((col[1] - green_avg)/green_deviation)
            z_score_blue.append((col[2] - blue_avg)/blue_deviation)

        # print("Averages red, green, blue: {}, {}, {}".format(red_avg, green_avg, blue_avg))
        # print("Deviation red green blue: {}, {}, {}".format(red_deviation, green_deviation, blue_deviation))

        # print("Z score for red: {}".format(z_score_red))
        # print("Z score for green: {}".format(z_score_green))
        # print("Z score for blue: {}".format(z_score_blue))
        filtered_color_list = []
        filtered_z_score_list = []
        for i in range(len(color_list)):
            if z_score_red[i] < 1.5 and z_score_red[i] > -1.5 and z_score_green[i] < 1.5 and z_score_green[i] > -1.5 and z_score_blue[i] < 1.5 and z_score_blue[i] > -1.5:
                filtered_color_list.append(color_list[i])
                filtered_z_score_list.append((z_score_red[i], z_score_green[i], z_score_blue[i]))


        # print("Filtered colors: {}".format(filtered_color_list))
        # print("Filtered z_score: {}".format(filtered_z_score_list))

        #average expected result
        averaged_expected_result_tensor = sum(filtered_color_list) / len(filtered_color_list)
        return averaged_expected_result_tensor
    
    def set_data(self, colors_for_image):
        self.colors_for_image = colors_for_image

    def get_filtered_averaged_result(self):
        return self.__Z_score_algorithm()