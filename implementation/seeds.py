class seedsGen(object):

    def __init__(self, file_loc):
        self.file_loc = file_loc
        self.seeds = self._fill_seeds()
        self.cur_seed_index = 0
        self.total_seeds = len(self.seeds)

    def _fill_seeds(self):
        with open(self.file_loc, 'r') as file_:
            lines = file_.readlines()
        output_list = []
        for line in lines:
            output_list.append(int(line))
        return output_list


    def get_single_seed(self):
        seed = self.seeds[self.cur_seed_index]
        self.cur_seed_index += 1
        if self.cur_seed_index >= self.total_seeds:
            self.cur_seed_index = 0
        return seed

    def get_batch_of_seeds(self, no_seeds):
        batch_of_seeds = self.seeds[self.cur_seed_index:self.cur_seed_index + no_seeds]
        self.cur_seed_index += no_seeds
        return batch_of_seeds
