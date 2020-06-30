import random, string
import dynet as dy 
import numpy as np
import utils


def main():
    data = utils.get_data()
    random.shuffle(data)
    data = data[:512]
    vocab = list(set([word for p in data for word in p]))

    # haiku_data = [poem for poem in data if poem_is_haiku(poem)]
    # haiku_vocab = list(set([word for p in haiku_data for word in p]))
    poet = Poet(vocab, utils.START_WORD, utils.DELIM, utils.N_EMBEDDINGS, utils.N_HIDDEN)
    
    
    # n_epoch = len(data) / utils.BATCH_SIZE
    n_epoch = 200
    print('len(data)={}, n_epoch={}'.format(len(data), int(n_epoch)))
    for i in range(int(n_epoch)):
        loss = poet.train(data)
        print('epoch {}, loss={}\n'.format(i, loss))
        # if i % 10 == 0:
        #     print('saving...')
        #     poet.save_parameters(i, loss)
        for _ in range(1):
            haiku = poet.generate_haiku()
            if haiku: utils.print_poem(haiku)
            else: utils.print_poem(poet.generate_unrestricted_poem())

    for _ in range(10):
        haiku = poet.generate_haiku()
        if haiku: utils.print_poem(haiku)
        else: utils.print_poem(poet.generate_unrestricted_poem())


class Poet():
    def __init__(self, vocab, start_word, delim, num_inputs, num_hidden):
        self.words = vocab
        self.start_word = start_word
        self.delim = delim

        self.w2i = dict([(word, i) for (i, word) in enumerate(self.words)])
        self.i2w = dict([(i, word) for (i, word) in enumerate(self.words)])

        # construct LSTM model
        num_layers = 2
        self.model = dy.Model()
        self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=.15)
        self.builder = dy.LSTMBuilder(num_layers, num_inputs, num_hidden, self.model)

        num_words = len(self.words)
        self.embs = self.model.add_lookup_parameters((num_words, num_inputs))
        self.w1 = self.model.add_parameters((num_words, num_hidden))
        self.b1 = self.model.add_parameters((num_words))

        # populate embs from glove or from file
        self.embs.init_from_array(utils.get_glove_vectors(self.words))
        # self.model.populate('./saves4/200_21.338068589170675.dy.model')
    
    
    def train(self, data):
        random.shuffle(data)

        total_loss = 0
        for i in range(0, len(data), utils.BATCH_SIZE):
            dy.renew_cg()
            batch = data[i:i+utils.BATCH_SIZE]
            random.shuffle(batch)

            batch_errs = []
            for haiku in batch:
                batch_errs += self._build_lm_graph([self.w2i[word] for word in haiku])
            batch_loss = dy.esum(batch_errs)
            batch_loss.backward()
            self.trainer.update()
            total_loss += batch_loss.scalar_value() / utils.BATCH_SIZE
        return total_loss / (len(data) / utils.BATCH_SIZE)
    
    def generate_haiku(self):
        dy.renew_cg()
        max_tries = 50
        for i in range(max_tries):
            line1 = [self.w2i[self.start_word]] + self._generate_line([self.w2i[self.start_word]])
            if utils.nsyl_line(self._convert_ids_to_words(line1)) == 5: break
        max_tries -= i
        for i in range(max_tries):
            line2 = self._generate_line(line1)
            if utils.nsyl_line(self._convert_ids_to_words(line2)) == 7: break
        max_tries -= i
        for i in range(max_tries):
            line3 = self._generate_line(line1+line2)
            if utils.nsyl_line(self._convert_ids_to_words(line3)) == 5: break    

        poem = self._convert_ids_to_words(line1 + line2 + line3)
        if utils.poem_is_haiku(poem):
            return poem
        else:
            return []

    def generate_unrestricted_poem(self):
        dy.renew_cg()
        state = self.builder.initial_state()
        curr_word = self.w2i[self.start_word]
        poem = [curr_word]
        num_lines = 0
        while True:
            x_t = self.embs[curr_word]
            state = state.add_input(x_t)
            next_word = self._next_word(state)
            poem.append(next_word)
            curr_word = next_word
            if curr_word == self.w2i[self.delim]: num_lines +=1
            if num_lines > 2: break
            if len(poem) > 25: break
        return self._convert_ids_to_words(poem)

    def _generate_line(self, features):
        state = self.builder.initial_state()
        for f in features:
            state = state.add_input(self.embs[f])

        line = []
        while True:
            curr_word = self._next_word(state)
            line.append(curr_word)

            if curr_word == self.w2i[self.delim]: break
            if len(line) > 12: break

            x_t = self.embs[curr_word]
            state = state.add_input(x_t)
        return line

    def _next_word(self, state):
        scores = self._get_scores(state)
        dist = dy.softmax(scores).vec_value()
        rnd = random.random()
        for i,p in enumerate(dist):
            rnd -= p
            if rnd <= 0: break 
        return i
    
    def _build_lm_graph(self, sent):
        state = self.builder.initial_state()
        errs = []
        for (cw,nw) in zip(sent,sent[1:]):
            emb = dy.lookup(self.embs, cw)
            state = state.add_input(emb)
            scores = self._get_scores(state)
            errs.append(dy.pickneglogsoftmax(scores, nw))
        return errs
    
    def _get_scores(self, state):
        layer_1 = state.output()
        output = self.b1 + (self.w1 * layer_1)
        return output
    
    def _convert_ids_to_words(self, ids):
        return [self.i2w[c] for c in ids]
    
    def save_parameters(self, epoch, uq):
        self.model.save('./saves4/{}_{}.dy.model'.format(epoch, str(uq)))

if __name__ == '__main__':
    main()
