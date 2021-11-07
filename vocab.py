class Vocabulary:

    def __init__(self, max_size):
        self.max_size = max_size
        self.current_size = 0
        self.table = dict()

    def add(self, word) -> int:
        if word not in self.table and not self.is_full():
            word_index = self.current_size
            self.table[word] = word_index
            self.current_size += 1
            return word_index

        elif word in self.table:
            word_index = self.table[word]
            return word_index
        else:
            return -1
                
    def is_full(self) -> bool:
        return self.current_size == self.max_size
    
    def __getitem__(self, word: str):
        if word in self.table:
            word_index = self.table[word] 
            return word_index
        return -1