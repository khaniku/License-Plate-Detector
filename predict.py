import segment_characters
import pickle
if __name__ == '__main__':
    print("Loading model")
    filename = './model.sav'
    model = pickle.load(open(filename, 'rb'))

    print('Model loaded. Predicting characters of number plate')
    classification_result = []
    for each_character in segment_characters.characters:
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1)
        result = model.predict(each_character)
        classification_result.append(result)

    print('Classification result')
    print(classification_result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    # print('Predicted license plate')
    # print(plate_string)

    column_list_copy = segment_characters.column_list[:]
    segment_characters.column_list.sort()
    rightplate_string = ''
    for each in segment_characters.column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    print('License plate')
    print(rightplate_string)