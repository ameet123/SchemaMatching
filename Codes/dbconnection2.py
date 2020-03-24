import pickle

import psycopg2 as db
import psycopg2.extras

import FeatureExtraction as fe

FLOATISH_TYPES = (700, 701, 1700)  # real, float8, numeric - 0
INT_TYPES = (20, 21, 23)  # bigint, int, smallint - 1
CHAR_TYPES = (25, 1042, 1043,)  # text, char, varchar   - 2
BOOLEAN_TYPES = (16,)  # bool                  - 3
DATE_TYPES = (1082, 1114, 1184,)  # date, timestamp, timestamptz - 4
TIME_TYPES = (1083, 1114, 1184, 1266,)  # time, timestamp, timestamptz, timetz - 5
keys = ['PRIMARY KEY', 'FOREIGN KEY']

features = ['type', 'length', 'key', 'unique', 'not_null', 'avgUsedLength', 'VarofLength', 'VarCoeffLength', 'average',
            'variance', 'Coeff', 'min', 'max', 'whitespace', 'specialchar', 'num2all', 'char2all', 'backslash',
            'brackets', 'hyphen']

INFO = True
DEBUG = False


def log(log_str):
    if INFO:
        print log_str


def debug(log_str):
    if DEBUG:
        print log_str


# run_type = "test"


class dbFeatureGeneration:
    def __init__(self, run_type):
        self.run_type = run_type

    def process(self):
        global table_id, typ, FILE_TYPE, n_data
        print "Running %s -> Matching pickle files to be created..." % self.run_type

        if self.run_type == "test":
            FILE_TYPE = "Match"  # Train or Match
            n_data = 52
            typ = 'ts_'
            table_id = 0
        elif self.run_type == "train":
            n_data = 1644
            typ = 'tr_'
            FILE_TYPE = "Train"
            table_id = 1

        conn = db.connect("dbname='postgres' user='postgres'")
        curs = conn.cursor()

        curs.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' ;")

        tab = curs.fetchall()
        print "Tables: %s" % tab

        table_nam = ''.join(tab[table_id])
        print " Table names --%s" % table_nam

        curs = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

        curs.execute('Select * from ' + table_nam + ' limit 20;')

        descr = curs.description  # description about the column attributes

        attributes = {}
        for i in range(0, len(descr)):
            attributes[descr[i][0]] = []

            if descr[i][1] in FLOATISH_TYPES:
                attributes[descr[i][0]].append(0)  # storing type - real/float/numeric of attribute

            if descr[i][1] in INT_TYPES:
                attributes[descr[i][0]].append(1)  # storing type - int of attribute

            if descr[i][1] in CHAR_TYPES:
                attributes[descr[i][0]].append(2)  # storing type - char of attribute

            if descr[i][1] in BOOLEAN_TYPES:
                attributes[descr[i][0]].append(3)  # storing type - boolean of attribute

            if descr[i][1] in DATE_TYPES:
                attributes[descr[i][0]].append(4)  # storing type - date of attribute

            if descr[i][1] in TIME_TYPES:
                attributes[descr[i][0]].append(5)  # storing type - time of attribute

            attributes[descr[i][0]].append(descr[i][3])  # storing length of attribute

        debug("Printing attributes ---")
        for key, val in attributes.items():
            debug("%s=>%s" % (key, val))

        # This command is giving the list of constraints - Keys, Unique, check on all the columns of a table.
        curs.execute(
            "SELECT tc.constraint_type, tc.table_name, kcu.column_name " +
            "FROM information_schema.table_constraints tc " +
            "LEFT JOIN information_schema.key_column_usage kcu " +
            "ON tc.constraint_catalog = kcu.constraint_catalog " +
            "AND tc.constraint_schema = kcu.constraint_schema " +
            "AND tc.constraint_name = kcu.constraint_name " +
            "WHERE tc.table_name='" + table_nam + "';")

        modifiers_column = curs.fetchall()

        # This command is extracting keys and unique from the above list.
        constraint_column = []
        for i in range(0, len(modifiers_column)):
            if modifiers_column[i][0] == 'PRIMARY KEY' or modifiers_column[i][0] == 'FOREIGN KEY' or \
                    modifiers_column[i][
                        0] == 'UNIQUE':
                constraint_column.append(modifiers_column[i])

        debug("Constraint columns ---")
        debug(constraint_column)

        # Append 0 for Primary and Foreign Key , then update as we find the column in the list.
        for i in range(0, len(descr)):
            attributes[descr[i][0]].append(0)

        for i in range(0, len(constraint_column)):
            if constraint_column[i][0] in keys:
                attributes[constraint_column[i][2]][-1] = 1  # storing Primary / Foreign Key of attribute

        # Append 0 for Unique Constraint , then update as we find the column in the list.
        for i in range(0, len(descr)):
            attributes[descr[i][0]].append(0)

        for i in range(0, len(constraint_column)):
            if constraint_column[i][0] == 'UNIQUE':
                attributes[constraint_column[i][2]][-1] = 1  # storing Unique constraint of attribute

        # This command is giving the list of nullable constraints on all the columns of a table.
        curs.execute(
            "select column_name, IS_NULLABLE from INFORMATION_SCHEMA.COLUMNS where table_name ='" + table_nam + "';")
        Null_attributes = curs.fetchall()

        # Append 0 for Not_Null attribute, then update as we find the column in the list.
        for i in range(0, len(descr)):
            attributes[descr[i][0]].append(0)

        for i in range(0, len(Null_attributes)):
            if Null_attributes[i][1] == 'NO':
                attributes[Null_attributes[i][0]][
                    -1] = 1  # storing Not-null constraint of attribute   - it cannot be null

        curs = conn.cursor()

        for i in range(0, len(descr)):
            curs.execute('Select ' + descr[i][0] + ' from ' + table_nam + ';')
            # data = curs.fetchall()
            data = list(zip(*curs.fetchall())[0])
            # print type(data)
            temp = []
            # Converting the date format to mm/dd/yyyy
            if attributes[descr[i][0]][0] == 4:
                for j in range(0, len(data)):
                    temp.append(data[j].strftime('%m/%d/%Y'))

                data = temp
            fixed_length = attributes[descr[i][0]][1]
            attributes[descr[i][0]].append(float(fe.averageusedlength(data, n_data, fixed_length)))
            debug('average used length - %.2f' % fe.averageusedlength(data, n_data, fixed_length))

            attributes[descr[i][0]].append(fe.varianceoflength(data, fixed_length))
            attributes[descr[i][0]].append(fe.varianceCoefflength(data))
            if attributes[descr[i][0]][0] == 1 or attributes[descr[i][0]][0] == 0:
                number_features = list(fe.numFeatures(data, fixed_length, n_data))
                for values in number_features:
                    attributes[descr[i][0]].append(values)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
            else:
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(0.0)
                attributes[descr[i][0]].append(fe.WhiteSpaceFeature(data))
                attributes[descr[i][0]].append(fe.specialChars(data))
                attributes[descr[i][0]].append(fe.NumbertoAll(data))
                attributes[descr[i][0]].append(fe.ChartoAll(data))
                attributes[descr[i][0]].append(fe.Numberofbackslash(data))
                attributes[descr[i][0]].append(fe.Numberofbrackets(data))
                attributes[descr[i][0]].append(fe.Numberofhyphen(data))

        debug("Attributes of state-----")
        debug(attributes['state'])
        debug("END attrib state ------")

        new_attributes = {}
        for k in attributes.keys():
            key_val = typ + k
            new_attributes[key_val] = attributes[k]

        file_name = "../Feature_Vectors/DataFeatures_" + FILE_TYPE + ".pickle"
        log("Saving features file:%s"%file_name)
        pickle.dump(new_attributes, open(file_name, 'wb'))

        print("Total attributes:%d" % len(new_attributes))


if __name__ == '__main__':
    for run_type in ["train", "test"]:
        a = dbFeatureGeneration(run_type)
        a.process()
