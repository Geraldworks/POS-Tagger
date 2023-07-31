# global variables to maintain tag and word counts
unique_words = []
tag_to_word_mappings = {}
tag_counts = {}

# smoothing constants
SMOOTHING_CONSTANTS = [0.01, 0.1, 1, 10]

# Main process function
def process(training_filename):
    with open(training_filename, "r") as f:
        for line in f:
            contents = line.split()

            if len(contents) == 0:
                continue
            
            # convert all words to lower case
            word = contents[0].lower()
            tag = contents[1]

            # cleaning up for users
            # if "user" in word:
            #     word = "user"

            # Create a mapping of tag -> word -> counts
            if tag not in tag_to_word_mappings:
                tag_to_word_mappings[tag] = {}

            # Keeps track of the number of times each tag is observed in the training file
            if tag not in tag_counts:
                tag_counts[tag] = 1
            else:
                tag_counts[tag] += 1
            
            # If the word is not in the word -> counts mapping, add the word in
            if word not in tag_to_word_mappings[tag]:
                tag_to_word_mappings[tag][word] = 1
            # If the word is in the word -> counts mapping, increment the count by 1
            else:
                tag_to_word_mappings[tag][word] += 1

            # Keep track of all the unique word counts in the training set
            if word not in unique_words:
                unique_words.append(word)

    # Creates the file
    with open("naive_output_probs.txt", "w") as f:
        for tag, dic in tag_to_word_mappings.items():
            total_tag_count = tag_counts[tag] # Retrieves from the dictionary
            for word, count in dic.items():
                f.write(f"{(tag, word, count/total_tag_count, count, total_tag_count)} \n")
                # (tag, word, probability, tag -> word counts, total tag counts)

# Q2 and Q3
def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    smooth = SMOOTHING_CONSTANTS[1] # the smooth constant to use
    best_word_to_tag_smoothed_prob = {} # stores a dictionary of word -> probability vector (tag -> prob)

    # Get all unique tags
    twitter_tags = []
    with open("twitter_tags.txt", "r") as tags_txt:
        lines = tags_txt.readlines()
        twitter_tags = list(map(lambda x: x[:-1], lines))

    # Get smoothed probability for non-existent word-tag combinations
    # This is to assign an initial probability of all tags, for any word
    probs_for_non_existent = {}
    for tag in twitter_tags:
        probs_for_non_existent[tag] = (smooth) / (tag_counts[tag] + smooth * (len(unique_words) + 1))

    # this step populates the best_word_to_tag_smoothed_prob dictionary
    # the mapping is word -> probability vector (tag -> prob)
    with open(in_output_probs_filename) as f:
        for line in f:
            contents = eval(line)
            tag, word, _, tag_word_count, total_tag_count = contents

            smoothed_prob = (tag_word_count + smooth) / (total_tag_count + smooth * (len(unique_words) + 1))

            if word not in best_word_to_tag_smoothed_prob:
                # set an initial probability for all tags
                best_word_to_tag_smoothed_prob[word] = probs_for_non_existent.copy()
            # replace the tag probability associated with this word
            # there will always only have one unique mappings for this
            best_word_to_tag_smoothed_prob[word][tag] = smoothed_prob 

    # predicting
    with open(in_test_filename, 'r') as test, open(out_prediction_filename, 'w') as predict:
        for line in test:
            contents = line.split() # return array

            if len(contents) == 0:
                predict.write("\n") # account for line breaks
            else:
                word = contents[0].lower()

                # collapse all user IDs to one word
                # if "user" in word:
                #     word = "user"
                try:
                    # argmax j
                    word_probs = best_word_to_tag_smoothed_prob[word].items()
                    predicted_tag = list(filter(lambda x: x == max(word_probs, key=lambda x: x[1]), word_probs))[0][0]
                    predict.write(predicted_tag + "\n")
                # predict random tag if word is unseen
                except KeyError:
                    word_probs = probs_for_non_existent.items()
                    predicted_tag = list(filter(lambda x: x == max(word_probs, key=lambda x: x[1]), word_probs))[0][0]
                    predict.write(predicted_tag + "\n")
                    #predict.write(list(tag_counts.keys())[random.randint(0, 24)] + "\n")

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    smooth = SMOOTHING_CONSTANTS[1]
    
    # Word to tag probability is word -> (tag, probability)
    word_to_tag_probabilities = {}

    # Get all unique tags
    twitter_tags = []
    with open("twitter_tags.txt", "r") as tags_txt:
        lines = tags_txt.readlines()
        twitter_tags = list(map(lambda x: x[:-1], lines))

    # Get smoothed probability for non-existent word-tag combinations
    # This is to assign an initial probability of all tags, for any word
    probs_for_non_existent = {}
    for tag in twitter_tags:
        # P(x = w | y = j)
        smoothed_prob = (0 + smooth) / (tag_counts[tag] + smooth * (len(unique_words) + 1)) 
        # P(y = j)
        tag_prob = tag_counts[tag] / sum(tag_counts.values())
        # P(x = w | y = j) * P(y = j)
        probability = smoothed_prob * tag_prob 
        probs_for_non_existent[tag] = probability
    
    # processing
    with open(in_output_probs_filename) as f:
        for line in f:
            contents = eval(line)
            tag, word, _, tag_word_count, total_tag_count = contents

            # P(x = w|y = j)
            smoothed_prob = (tag_word_count + smooth) / (total_tag_count + smooth * (len(unique_words) + 1))

            # P(y = j)
            tag_prob = tag_counts[tag] / sum(tag_counts.values())
            
            # P(x = w | y = j) * P(y = j) which is proportional to P(y = j | x = w)
            probability = smoothed_prob * tag_prob

            if word not in word_to_tag_probabilities:
                # set an initial probability for all tags
                word_to_tag_probabilities[word] = probs_for_non_existent.copy()
            
            # replace the tag probability associated with this word
            # there will always only have one unique mappings for this
            word_to_tag_probabilities[word][tag] = probability
    
    # predicting
    with open(in_test_filename, 'r') as test, open(out_prediction_filename, 'w') as predict:
        for line in test:
            contents = line.split() # return array

            if len(contents) == 0:
                predict.write("\n")
            else:
                word = contents[0].lower()

                # collapse all user IDs to one word
                # if "user" in word:
                #     word = "user"
                try:
                    # argmax j
                    word_probs = word_to_tag_probabilities[word].items()
                    predicted_tag = list(filter(lambda x: x == max(word_probs, key=lambda x: x[1]), word_probs))[0][0]
                    predict.write(predicted_tag + "\n")
                # predict random tag if word is unseen
                except KeyError:
                    word_probs = probs_for_non_existent.items()
                    predicted_tag = list(filter(lambda x: x == max(word_probs, key=lambda x: x[1]), word_probs))[0][0]
                    predict.write(predicted_tag + "\n")
                    # predict.write(list(tag_counts.keys())[random.randint(0, 24)] + "\n")

                    from pprint import pprint

# Q4 and Q5
def create_transition_probabilities(training_file, all_tags_file, output_file_name):

    # Keeping track of all the counts of each tag
    tag_counts = {"START": 0, "STOP":0}
    # keeping track of all available tags
    tags = []

    # counts of seeing (i, j)
    transition_counts = {} 

    # transition probabilities
    transition_probs = {}

    with open(all_tags_file) as f:
        for line in f:
            # Retrieving the tag
            tag = line.split()[0]
            tags.append(tag)
            # Setting tag counts to 0
            tag_counts[line.split()[0]] = 0
            # Initialising aSTART,tag
            transition_counts[("START", tag)] = 0
            # Initialising atag,STOP
            transition_counts[(tag, "STOP")] = 0

    # creating all tag pairs
    for first_tag in tags:
        for second_tag in tags:
            transition_counts[(first_tag, second_tag)] = 0

    # start of creating transition probabillities
    details = [[]]
    with open(training_file) as f:
        for line in f:
            details.append(line.split())

    from_state = None
    to_state = None

    for i in range(len(details) - 1): 
        from_state = details[i]
        to_state = details[i+1]
        
        if not from_state: # Means this is START state
            tag_counts["START"] += 1
            transition_counts[("START", to_state[1])] += 1
        elif not to_state: # Means this is STOP state
            tag_counts[from_state[1]] += 1
            transition_counts[(from_state[1], "STOP")] += 1
        else:
            tag_counts[from_state[1]] += 1
            transition_counts[(from_state[1], to_state[1])] += 1

    for key, value in transition_counts.items(): # key is a pair of (from, to)
        from_state = key[0]
        transition_probs[key] = value / tag_counts[from_state]

    # writing to trans_probs.txt file
    with open(output_file_name, "w") as output:
        output.write(f"FROM TO TRANSITION_PROB \n")
        for key, prob in transition_probs.items():
            output.write(f"{key[0]} {key[1]} {prob}" + "\n")

    return transition_probs

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    
    # create the smoothed output_probs.txt file and viterbi_predictions.txt

    ### transition probability and trans_probs.txt file creation ###
    a = create_transition_probabilities("twitter_train.txt", in_tags_filename, "trans_probs.txt")

    # creating all tags
    all_states = []
    with open(in_tags_filename, "r") as tags_txt:
        lines = tags_txt.readlines()
        all_states = list(map(lambda x: x[:-1], lines))

    # mapping from index to tag
    number_to_state_mappings = dict(enumerate(all_states))

    # creating emission probabilities with smoothing
    smooth = SMOOTHING_CONSTANTS[1] 
    emission_probabilities = {}
    probs_for_non_existent = {}

    # handling unseen words
    for tag in all_states:
        probs_for_non_existent[tag] = (smooth) / (tag_counts[tag] + smooth * (len(unique_words) + 1))

    # retrieving emission probabilities for seen words
    with open('naive_output_probs.txt', 'r') as f:
        for line in f:
            contents = eval(line)
            tag, word, _, tag_word_count, total_tag_count = contents
            smoothed_prob = (tag_word_count + smooth) / (total_tag_count + smooth * (len(unique_words) + 1))
            emission_probabilities[(tag, word)] = smoothed_prob 
    
    b = emission_probabilities

    ### saving to output_probs.txt ###
    with open("output_probs.txt", "w") as f:
        f.write("TAG WORD EMISSION_PROB \n")
        for tup, prob in emission_probabilities.items():
            f.write(f"{tup[0]} {tup[1]} {prob} \n")

    def get_tag_sequence(bp_matrix, final_bp):
        tag_seq = []
        bp_length = len(bp_matrix) - 1
        while final_bp != "*":
            tag_seq.append(number_to_state_mappings[final_bp])
            final_bp = bp_matrix[bp_length][final_bp]
            bp_length -= 1

        return tag_seq[::-1]

    def vit(sentence_as_list):
        """
        Returns a list of tags for the given input sentence
        """

        pi = [[0 for i in range(len(all_states))] for j in range(len(sentence_as_list))]
        bp = [[0 for i in range(len(all_states))] for j in range(len(sentence_as_list))]

        # create first word probabilities 
        first_row = pi[0]
        bp[0] = list(map(lambda x: "*", bp[0]))

        for index, tag in number_to_state_mappings.items():
            # check if emission probability exists
            if (tag, sentence_as_list[0]) in b:
                first_row[index] = a[("START", tag)] * b[(tag, sentence_as_list[0])]
            else:
                first_row[index] = a[("START", tag)] * probs_for_non_existent[tag]

        for i in range(1, len(sentence_as_list)): # looping over every word
            for next_index, next_tag in number_to_state_mappings.items(): # looping over every next tag
                pi_k_v = []
                for curr_index, curr_tag in number_to_state_mappings.items(): # looping over every current tag
                    if (next_tag, sentence_as_list[i]) in b:
                        pi_k_v.append(pi[i-1][curr_index] * a[(curr_tag, next_tag)] * b[(next_tag, sentence_as_list[i])])
                    else:
                        pi_k_v.append(pi[i-1][curr_index] * a[(curr_tag, next_tag)] * probs_for_non_existent[next_tag])

                best_prob = max(pi_k_v)
                best_index = pi_k_v.index(best_prob)

                pi[i][next_index] = best_prob
                bp[i][next_index] = best_index

        # get the max prob and final_bp
        final_iteration_probs = []
        for next_index, next_tag in number_to_state_mappings.items():
            final_iteration_probs.append(pi[-1][next_index] * a[(next_tag, "STOP")])

        final_max_prob = max(final_iteration_probs)
        final_bp = final_iteration_probs.index(final_max_prob)

        # output the list of predicted tags
        return get_tag_sequence(bp, final_bp)
    
    ### predicting the test words and output viterbi_predictions.txt ###
    with open(in_test_filename, "r") as f, open(out_predictions_filename, 'w') as predict:
        curr_sentence = []
        is_complete_sentence = False
        for line in f:
            contents = line.split()
            if len(contents) != 0:
                word = contents[0].lower()
                curr_sentence.append(word)
            else:
                is_complete_sentence = True
            
            # when there is a complete sentence
            if is_complete_sentence:
                output_tags = vit(curr_sentence)
                # writing to file
                for tag in output_tags: 
                    predict.write(f"{tag} \n")
                predict.write("\n")
                # reset tracking variables
                curr_sentence = []
                is_complete_sentence = False

'''
(1) Emission probabilities are encoded as P(Xt | Yt, Yt-1) instead of P(Xt | Yt) 
    as a previous tag could affect the current word. We removed the assumption
    that Xt is independent of Yt-1 given Yt to better model POS tagging.

(2) We collapsed all "@USER_xxxxxx" to "@USER_" as user handles are unique and 
    can overcomplicate prediction for @. Based on Twitter's conventions, all
    usernames have the format "@USER_xxxxxx", and the POS tag of a user ID is 
    independent of the specific identifier. By clustering user IDs as such, we
    allow our model to better learn the POS of user IDs and better predict POS
    of user IDs (as unseen user IDs will no longer be classified as unseen).

(3) Similarly, we collapsed all "https://...", "http://..." and "www...." to 
    "https" as URLS can be unique and can overcomplicate prediction for U.

(4) Similarly, we collpased all "#..." to "#" as hashtags can be classified 
    as a single type of word.
'''
def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename, out_predictions_filename):
    smooth = SMOOTHING_CONSTANTS[1]
    
    # list of unique words
    unique_words_list = []

    # a function to create the new emission probabilities P(Xt | Yt, Yt-1)
    def preprocess_second_order(training_filename):
        """
        counts is a dictionary that stores (Yt-1, Yt) -> count of seeing this pair
        if Yt-1 does not exist, it will be stored as (*, Yt) -> counts instead
        """
        tracking_pairs = {}
        prev_tags = []
        curr_index = -1
        
        # keeping track of the counts for the current tag and previous tag to the word
        with open(training_filename, "r") as f:
            for line in f:
                contents = line.split()

                # when it's a new line, it's a new sentence
                if len(contents) == 0:
                    curr_index = -1
                    prev_tags = []
                    continue
                
                word, tag = contents[0].lower(), contents[1]
                if "@user_" in word:
                    word = "@user_"
                if "http://" in word or "https://" in word or "www." in word:
                    word = "http://"
                if "#" in word:
                    word = "#"
                
                if curr_index == -1:
                    tags = ("*", tag)
                    if tags not in tracking_pairs: 
                        tracking_pairs[tags] = {}
                        tracking_pairs[tags][word] = 1
                    else:
                        if word not in tracking_pairs[tags]:
                            tracking_pairs[tags][word] = 1
                        else:
                            tracking_pairs[tags][word] += 1
                    
                else: 
                    tags = (prev_tags[curr_index], tag)
                    if tags not in tracking_pairs: 
                        tracking_pairs[tags] = {}
                        tracking_pairs[tags][word] = 1
                    else:
                        if word not in tracking_pairs[tags]:
                            tracking_pairs[tags][word] = 1
                        else:
                            tracking_pairs[tags][word] += 1
                
                # Keep track of all the unique word list in the training set
                if word not in unique_words_list:
                    unique_words_list.append(word)
                
                prev_tags.append(tag)
                curr_index += 1

        smoothed_probabilities = {}

        # smoothing
        for tags, d in tracking_pairs.items():
            tag_pair_count = sum(d.values())
            smoothed_probabilities[tags] = {}
            for word, count in d.items():
                smoothed_probabilities[tags][word] = (count + smooth) / (tag_pair_count + smooth * (len(unique_words_list) + 1))
        
        return smoothed_probabilities

    # getting emission probabilities
    b = preprocess_second_order("twitter_train.txt")
    
    # obtains all tags
    all_states = []
    with open(in_tags_filename, "r") as tags_txt:
        lines = tags_txt.readlines()
        all_states = list(map(lambda x: x[:-1], lines))
    
    # smoothing for unseen words
    probs_for_non_existent = {}
    all_states_with_starting_state = all_states.copy()
    all_states_with_starting_state.append("*")

    for i in all_states_with_starting_state:
        for j in all_states_with_starting_state:
            if j != "*":
                probs_for_non_existent[(i, j)] = 0

    for pair in probs_for_non_existent.keys():
        # if word is unseen
        if pair in b:
            probs_for_non_existent[pair] = (smooth) / (sum(b[pair].values()) + smooth * (len(unique_words_list) + 1))
        # is pair is unseen
        else:
            probs_for_non_existent[pair] = (smooth) / (smooth * (len(unique_words_list) + 1))
    
    ### transition probability and trans_probs2.txt file creation ###
    a = create_transition_probabilities("twitter_train.txt", in_tags_filename, "trans_probs2.txt")

    # mapping from index to tag
    number_to_state_mappings = dict(enumerate(all_states))

    ### saving to output_probs2.txt using P(Xt | Yt, Yt-1) ###
    with open("output_probs2.txt", "w") as f:
        f.write("PREVIOUS_TAG CURRENT_TAG {WORD : EMISSION_PROB} \n")
        for tup, prob in b.items():
            f.write(f"{tup[0]} {tup[1]} {prob} \n")

    def get_tag_sequence(bp_matrix, final_bp):
        tag_seq = []
        bp_length = len(bp_matrix) - 1
        while final_bp != "*":
            tag_seq.append(number_to_state_mappings[final_bp])
            final_bp = bp_matrix[bp_length][final_bp]
            bp_length -= 1

        return tag_seq[::-1]

    def vit(sentence_as_list):
        """
        Returns a list of tags for the given input sentence
        """

        pi = [[0 for i in range(len(all_states))] for j in range(len(sentence_as_list))]
        bp = [[0 for i in range(len(all_states))] for j in range(len(sentence_as_list))]

        # create first word probabilities 
        first_row = pi[0]
        bp[0] = list(map(lambda x: "*", bp[0]))

        for index, tag in number_to_state_mappings.items():
            # check if emission probability exists
            if ("*" , tag) in b:
                if word in b[("*", tag)]:
                    first_row[index] = a[("START", tag)] * b[("*", tag)][word]
                else:
                    first_row[index] = a[("START", tag)] * probs_for_non_existent[("*", tag)]
            else:
                first_row[index] = a[("START", tag)] * probs_for_non_existent[("*", tag)] # need smoothing here

        for i in range(1, len(sentence_as_list)): # looping over every word
            for next_index, next_tag in number_to_state_mappings.items(): # looping over every next tag
                pi_k_v = []
                for curr_index, curr_tag in number_to_state_mappings.items(): # looping over every current tag
                    if (curr_tag, next_tag) in b:
                        if sentence_as_list[i] in b[(curr_tag, next_tag)]:
                            pi_k_v.append(pi[i-1][curr_index] * a[(curr_tag, next_tag)] * b[(curr_tag, next_tag)][sentence_as_list[i]])
                        else:
                            pi_k_v.append(pi[i-1][curr_index] * a[(curr_tag, next_tag)] * probs_for_non_existent[(curr_tag, next_tag)])
                    else:
                        pi_k_v.append(pi[i-1][curr_index] * a[(curr_tag, next_tag)] * probs_for_non_existent[(curr_tag, next_tag)])

                best_prob = max(pi_k_v)
                best_index = pi_k_v.index(best_prob)

                pi[i][next_index] = best_prob
                bp[i][next_index] = best_index

        # get the max prob and final_bp
        final_iteration_probs = []
        for next_index, next_tag in number_to_state_mappings.items():
            final_iteration_probs.append(pi[-1][next_index] * a[(next_tag, "STOP")])

        final_max_prob = max(final_iteration_probs)
        final_bp = final_iteration_probs.index(final_max_prob)

        # output the list of predicted tags
        return get_tag_sequence(bp, final_bp)
    
    ### predicting the test words and output viterbi_predictions.txt ###
    with open(in_test_filename, "r") as f, open(out_predictions_filename, 'w') as predict:
        curr_sentence = []
        is_complete_sentence = False
        for line in f:
            contents = line.split()
            if len(contents) != 0:
                word = contents[0].lower()
                if "@user_" in word:
                    word = "@user_"
                if "http://" in word or "https://" in word or "www." in word:
                    word = "http://"
                if "#" in word:
                    word = "#"
                curr_sentence.append(word)
            else:
                is_complete_sentence = True
            
            # when there is a complete sentence
            if is_complete_sentence:
                output_tags = vit(curr_sentence)
                # writing to file
                for tag in output_tags: 
                    predict.write(f"{tag} \n")
                predict.write("\n")
                # reset tracking variables
                curr_sentence = []
                is_complete_sentence = False

def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)

def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = '.' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                   viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    
if __name__ == '__main__':
    process("twitter_train.txt")
    run()
