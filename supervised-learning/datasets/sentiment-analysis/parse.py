import csv

csv_file = "twitter-og.csv"
txt_file = "twitter-tweets.txt"

ind = 0
with open(txt_file, "w") as my_output_file:
    with open(csv_file, "r", encoding="ISO-8859-1") as my_input_file:
        for row in csv.reader(my_input_file):
            if ind > 5000:
                break

            row[0] = str(int(row[0]) // 4)
            my_output_file.write(" ".join(reversed(row))+'\n')

            ind += 1
    my_output_file.close()
