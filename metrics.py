def print_report(y_actual, y_pred):
    sentences_total = len(y_actual)
    words_total = sum([len(output) for output in y_actual])

    sentences_correct = 0
    words_correct = 0

    # variables to track information about sentences fully correctly predicted

    """
    Stores sum of length of all sentences predicted fully correctly
    """
    words_correct_all = 0

    """
    Stores length of longest sentence predicted fully correctly
    """
    longest_correct = 0

    for ya, yp in zip(y_actual, y_pred):
        all_correct = True
        for wya, wyp in zip(ya, yp):
            if wya == wyp:
                words_correct += 1
            else:
                all_correct = False
        if all_correct:
            sentences_correct += 1
            words_correct_all += len(ya)
            if len(ya) > longest_correct:
                longest_correct = len(ya)

    print('Sentences correct: {:.2f}%'.format(sentences_correct/sentences_total*100))
    print('Words correct: {:.2f}%'.format(words_correct/words_total*100))

    print('Average length of sentence: {:.2f}'.format(words_total/sentences_total))
    print('Average length of sentence correctly predicted: {:.2f}'.format(
        words_correct_all/sentences_total))
    print('Length of longest sentence correctly predicted: {}'.format(longest_correct))
