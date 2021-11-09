import torch
 
class UnigramTable:

    def __init__(self, max_size, 
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        
        self.max_size = max_size
        self.current_size = 0
        self.table = torch.zeros(int(self.max_size))
        self.table.to(self.device)
        self.weight_sum = 0
        
    
    def sample(self, rand):
        rand_num = int(rand.uniform(0, self.current_size))
        output = self.table[rand_num]
        return output

    def update(self, word_index, weight, rand):
        self.weight_sum += weight
        if self.current_size < self.max_size:
            new_size = min(rand.round(weight) + self.current_size, self.max_size)
            self.table[self.current_size: new_size] = word_index
            self.current_size = new_size            
        else:
            n = rand.round(weight / self.weight_sum) * self.max_size
            self.table[0:n] = word_index
    
    def __str__(self):
        return self.table.__str__()