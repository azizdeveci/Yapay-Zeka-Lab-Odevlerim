import torch
import torch.nn as nn
import torch.optim as optim
from collections import Counter
from itertools import product

# Veri hazırlama
text = """Bu ürün beklentimi fazlasıyla karşıladı, kesinlikle tavsiye ederim!
Kalitesi ve fiyat performans oranı için mükemmel bir seçim.
Hızlı kargo ve sorunsuz ürün, teşekkürler.
Kullanımı çok kolay, günlük hayatımı oldukça kolaylaştırdı.
Ürünün tasarımı ve işlevselliği beni gerçekten etkiledi."""

words = text.replace(",", "").replace(".", "").replace("!", "").lower().split()
word_counts = Counter(words)
vocab = sorted(word_counts, key=word_counts.get, reverse=True)
word_to_index = {word: i for i, word in enumerate(vocab)}
ix_to_word = {i: word for i, word in enumerate(vocab)}
data = [(words[i], words[i+1]) for i in range(len(words)-1)]




class LSTM(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x):
        
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x.view(1, 1, -1))  # DÜZELTME: Sadece output'u al
        output = self.fc(lstm_out.view(1, -1))
        return output

# Hyperparameter tuning
def prepare_sequence(seq, to_ix):
    
    return torch.tensor([to_ix[w] for w in seq], dtype=torch.long)

embedding_size = [8, 16]

hidden_size = [32, 64]

learning_rate = [0.01, 0.005]

best_loss = float("inf")
best_params = {}

print("Hyperparameter tuning start...")

for emb_size, h_size, lr in product(embedding_size, hidden_size, learning_rate):
    print(f"Deneme: emb:{emb_size} hid size:{h_size} lr:{lr}")
    
    model = LSTM(len(vocab), emb_size, h_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    epochs = 50
    
    for epoch in range(epochs):
        
        epoch_loss = 0
        
        for word, next_word in data:
            
            model.zero_grad()
            input_tensor = prepare_sequence([word], word_to_index)
            target_tensor = prepare_sequence([next_word], word_to_index)
            output = model(input_tensor)
            loss = loss_function(output, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch:{epoch} Loss:{epoch_loss:.5f}")
    
    if epoch_loss < best_loss:
        best_loss = epoch_loss
        best_params = {
            "embedding_dim": emb_size,
            "hidden_dim": h_size,
            "learning_rate": lr
        }

print(f"Best params: {best_params}")
    

# en iyi parametrelerle eğitim

final_model=LSTM(len(vocab), best_params['embedding_dim'], best_params['hidden_dim'])

optimizer = optim.Adam(final_model.parameters(), lr = best_params['learning_rate'])


print("Final Model Training")

epochs = 100

for epoch in range(epochs):
    
    epoch_loss = 0
    
    for word, next_word in data:
        
        final_model.zero_grad()
        input_tensor = prepare_sequence([word], word_to_index)
        target_tensor = prepare_sequence([next_word], word_to_index)
        output = final_model(input_tensor)
        loss = loss_function(output, target_tensor)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if epoch % 10 == 0:
        print(f"Epoch:{epoch} Loss:{epoch_loss:.5f}")


## test ve değerlendirme

# kelme tahmini fonksiyonu

def predict_sequence(start_word, num_words):
    current_word = start_word # şu anki kelime başlangıç kelime olarak ayarlanır
    output_sequence = [current_word]
    
    for _ in range(num_words): # belritilen sayıda çıktı tahmini
        
        with torch.no_grad(): # gradyan hesaplaması olamdan
            
            input_tensor = prepare_sequence([current_word], word_to_index)# kelime tensore çevir
            output = final_model(input_tensor) 
            predicted_idx =torch.argmax(output).item() #en yüksek olasılığa sahiğ kelimenin indexi
            predicted_word = ix_to_word[predicted_idx]#indexe karşılık gelen kelşmeyi return eder 
            output_sequence.append(predicted_word) 
            current_word = predicted_word # bir sonraki tahmin için mevcut kelimeleri güncelle
    return  output_sequence

start_word="harika"
num_predictions=20
print("  ".join(predict_sequence(start_word, num_predictions)))

            

























