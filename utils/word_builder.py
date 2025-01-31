import sys

class WordBuilder:
    def __init__(self):
        self.cur_word = ''
        self.last_detected_letter = ''
        self.check_len = 3

    def getCurWord(self):
        return self.cur_word
    
    def addLetter(self, letter):
        self.last_detected_letter = letter
        if len(self.cur_word) > self.check_len:
            active_check_len = self.check_len
        else:
            active_check_len = len(self.cur_word)
        for i in range(active_check_len):
            if self.cur_word[-1-i] == letter:
                return self.cur_word
        self.cur_word += letter
        return self.cur_word

    def add_last_letter(self):
        self.cur_word += self.last_detected_letter
        return self.cur_word

    def backspace(self):
        self.cur_word = self.cur_word[:-1]
        return self.cur_word

    def erase(self):
        self.cur_word = ''
        return self.cur_word

    def addSpace(self):
        self.cur_word += ' '
        return self.cur_word
    
if __name__ == '__main__':
    builder = WordBuilder()
    _in = ''
    while _in != "0":
        _in = sys.stdin.readline().strip()
        print(builder.addLetter(_in))