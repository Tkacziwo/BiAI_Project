import os
import torch
import CNN
import brainTrainer as trainer
import DataFilter
from ImageLoader import ImageLoader as im
from ImageColorDataSet import ImageColorDataSet as icd

def rgb_to_tensor(rgb: str):

    red = int(rgb[0:2], 16)
    green = int(rgb[2:4], 16)
    blue = int(rgb[4:6], 16)

    t = torch.tensor([red, green, blue], dtype=torch.float32) / 255.0
    return t

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

print(f"Using {device} device")

#loading of previously saved "good" model if none, train again
brainModels = []
for modelName in os.listdir("models"):
    brain = CNN.CNN()
    path = "models/"+modelName
    brain.load_state_dict(torch.load(path))
    brainModels.append(brain)

if brainModels.__len__() == 0:
    print("No models found")
    image_tensor_loader = im()
    image_tensor_loader.load_input_photos_to_tensor("input-photos")

    loaded_images_tensors = image_tensor_loader.get_input_photo_tensors()
    if loaded_images_tensors.count == 0:
        print("Error while loading images")
        exit()
    else:
        print("Loaded all images!")

    print("Training brain... this may take a while...")

    # Using data filter to load expected results
    filter = DataFilter.DataFilter()
    colorsPath = os.path.join(os.path.curdir, 'expected-results')
    dataFilter = DataFilter.DataFilter()
    for filename in os.listdir(colorsPath):
        if "_Time" not in filename:
            dataFilter.loadColorAnnotations(os.path.join(colorsPath, filename))


    dataFilter.filterData()

    image_color_dataset = icd("input-photos", dataFilter.filteredData)
    filtered_brain = trainer.FilteredBrainTrainer(dataset= image_color_dataset,
                                                  device = device,
                                                  loss_function=torch.nn.MSELoss())
    
    filtered_brain.train_brain(20)

    image_color_dataset.switchOnTesting()

    brain = filtered_brain.get_model()
    brain.eval()
    for i in range(image_color_dataset.__len__()):
        img, expected_result = image_color_dataset.__getitem__(i)
        trained_output = brain(img)
        print("Trained output: {}. Correct result: {}".format(trained_output, expected_result))
        
    print("Brain trained. Run application again to verify model")

    image_color_dataset.save_dataset()
else:

    colorsPath = os.path.join(os.path.curdir, 'expected-results')
    dataFilter = DataFilter.DataFilter()
    for filename in os.listdir(colorsPath):
        if "_Time" not in filename:
            dataFilter.loadColorAnnotations(os.path.join(colorsPath, filename))


    dataFilter.filterData()

    image_color_dataset = icd()
    image_color_dataset.load_predetermined_dataset("CreatedDataSet", dataFilter.filteredData)

    last_model: int = brainModels.__len__() - 1
    trained_brain = brainModels[last_model]
    trained_brain.eval()

    # Test model
    image_color_dataset.switchOnTesting()
    print("\n Testing model using test data \n")
    for i in range(image_color_dataset.__len__()):
        img_tensor, expected_result = image_color_dataset.get_item(i, "CreatedDataSet/test")
        
        output = trained_brain(img_tensor)
        output_list = output[0].tolist()
        expected_result_list = expected_result[0].tolist()
        output_list = [f"{x:.4f}" for x in output_list]
        expected_result_list = [f"{x:.4f}" for x in expected_result_list]
        print("Trained output: {}. Correct result: {}.".format(list(map(str, output_list)), list(map(str, expected_result_list))))   
    
    image_color_dataset.switchOnValidating()
    print("\nTesting model using validation data \n")
    for i in range(image_color_dataset.__len__()):
        img_tensor, expected_result = image_color_dataset.get_item(i, "CreatedDataSet/validate")
        
        output = trained_brain(img_tensor)
        output_list = output[0].tolist()
        expected_result_list = expected_result[0].tolist()
        output_list = [f"{x:.4f}" for x in output_list]
        expected_result_list = [f"{x:.4f}" for x in expected_result_list]
        print("Trained output: {}. Correct result: {}.".format(list(map(str, output_list)), list(map(str, expected_result_list))))  

exit(0)