import torch
from argparse import ArgumentParser
from dataset import TextDataset
from model import TextGenerationModel

def sample_next_char(predicted_char, temperature):

    if temperature == 0:
        return predicted_char.argmax()

    distribution = predicted_char / temperature
    distribution = torch.softmax(distribution, dim=0)

    return torch.multinomial(distribution, 1)

def generate_sentence(sample, model, dataset, temperature, length):

    predicted_char = torch.Tensor([[sample[0]]]).long()
    generated_sequence = []
    with torch.no_grad():
        h_0_c_0 = None

        for i in range(1, length):
            predicted_seq, h_0_c_0 = model.forward(predicted_char, h_0_c_0)
            if i < len(sample):
                predicted_char[0, 0] = sample[i]
            else:
                predicted_char[0, 0] = sample_next_char(predicted_seq.squeeze(), temperature)
                generated_sequence.append(predicted_char.item())

    return dataset.convert_to_string(generated_sequence)

def speak(length):

    dataset = torch.load('./outputs/secondexperiment/saved_dataset.dataset')
    device = torch.device('cpu')
    model = TextGenerationModel(64, 90, dataset.vocab_size, 128, 2, 0,  device)
    model.load_state_dict(torch.load('./outputs/secondexperiment/saved_model.pt', map_location='cpu'))

    _input = open('sample_start.txt', 'r', encoding='utf-8').read()
    idxs = dataset.convert_from_string(_input)
    text = generate_sentence(idxs, model,dataset, 0.5, length)
    print(text)



if __name__ == '__main__':
    speak(200)