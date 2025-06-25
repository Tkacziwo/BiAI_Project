import torch


class ColorsForImage():
    def __init__(self):
        super().__init__()
        self.image_file_name: str
        self.colors_for_image = []
        self.image_tensor: torch.tensor
    
    def set_data(self, image_name:str, colors: list, tensor:torch.tensor):
        self.image_file_name = image_name
        self.colors_for_image = colors
        self.image_tensor = tensor

    def get_image_name(self):
        return self.image_file_name
    
    def get_colors_for_image(self):
        return self.colors_for_image
    
    def get_image_tensor(self):
        return self.image_tensor