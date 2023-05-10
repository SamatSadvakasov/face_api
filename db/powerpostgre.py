import psycopg2
from psycopg2.extensions import AsIs
import numpy as np


class PowerPost:
    def __init__(self, host, port, dbname, user, pwd):
        self.host = host
        self.port = port
        self.dbname = dbname
        self.user = user
        self.pwd = pwd
        # self.engine = create_engine('postgresql://'+self.user+':'+self.pwd+'@'+self.host+':'+str(self.port)+'/'+self.dbname,echo=False)


    def search_from_persons(self, unique_id):
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host = self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute("SELECT person_name FROM fr.persons WHERE unique_id = %(uid)s", {'uid': unique_id})
            # fetching result from database reponse
            blob = cur.fetchone()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return None
        finally:
            if conn is not None:
                conn.close()
        return blob


    def get_top_one_from_face_db(self, vector):
        try:
            # connect to the PostgresQL database
            # FLAGS.psql_server, FLAGS.psql_server_port, FLAGS.psql_db, FLAGS.psql_user, FLAGS.psql_user_pass
            conn = psycopg2.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.user,
                                    password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # convert vector to SQL-readable text
            vector_str = AsIs('CUBE(ARRAY[' + str(vector.tolist()).strip('[|]') + '])')
            # SQL query
            select_query = """SELECT fr.faces.unique_id, fr.faces.vector,
                        (%(vector)s <-> fr.faces.vector) as distance
                        FROM fr.faces
                        ORDER BY %(vector)s <-> vector
                        ASC LIMIT 1"""
            # execute statement
            cur.execute(select_query, {'vector': vector_str})
            # fetch all results from database response
            result = cur.fetchall()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return []
        finally:
            if conn is not None:
                conn.close()
        return result


    def one_to_one(self, vector, person_id):
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.user,
                                    password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # convert vector to SQL-readable text
            vector_str = AsIs('CUBE(ARRAY[' + str(vector.tolist()).strip('[|]') + '])')
            # select statement
            select_query = """SELECT fr.faces.unique_id, fr.faces.vector,
                        (%(vector)s <-> fr.faces.vector) as distance
                        FROM fr.faces
                        WHERE unique_id=%(uid)s"""
            # execute statement
            cur.execute(select_query, {'vector': vector_str, 'uid': person_id})
            # fetch all results from database response
            result = cur.fetchall()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return []
        finally:
            if conn is not None:
                conn.close()
        return result


    def insert_new_person(self, unique_id, vector, person_name, person_surname, person_secondname, create_time, group_id, person_iin):
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host = self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # convert vector to SQL-readable text
            vector_str = AsIs('CUBE(ARRAY[' + str(vector.tolist()).strip('[|]') + '])')
            # execute the INSERT statement
            cur.execute("""
                        WITH 
                            faces_tbl AS
                            (INSERT INTO fr.faces (unique_id, vector) VALUES (%(uid)s, %(vector)s) RETURNING unique_id),
                            persons_tbl AS
                            (INSERT INTO fr.persons (unique_id, person_name, person_surname, person_secondname, insert_date, group_id, person_iin) 
                            VALUES (%(uid)s, %(p_name)s, %(p_surname)s, %(p_sname)s, %(c_time)s, %(g_id)s, %(p_iin)s) RETURNING unique_id)
                        SELECT faces_tbl.unique_id, persons_tbl.unique_id
                        FROM faces_tbl, persons_tbl;
                        """, {'uid': unique_id, 'vector': vector_str, 
                                'p_name': person_name, 'p_surname': person_surname, 
                                'p_sname': person_secondname, 'c_time': create_time, 
                                'g_id': group_id, 'p_iin': person_iin})
            # commit the changes to the database
            conn.commit()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return False
        finally:
            if conn is not None:
                conn.close()
        return True


    def insert_into_faces(self, unique_id, vector):
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host = self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # convert vector to SQL-readable text
            vector_str = AsIs('CUBE(ARRAY[' + str(vector.tolist()).strip('[|]') + '])')
            # execute the INSERT statement
            cur.execute("INSERT INTO fr.faces(unique_id, vector) VALUES (%(uid)s, %(vector)s)", {'uid': unique_id, 'vector': vector_str})
            # commit the changes to the database
            conn.commit()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return False
        finally:
            if conn is not None:
                conn.close()
        return True


    def delete_from_faces(self, unique_id):
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host = self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute("DELETE FROM fr.faces WHERE unique_id = %(uid)s", {'uid': unique_id})
            # commit the changes to the database
            conn.commit()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return False
        finally:
            if conn is not None:
                conn.close()
        return True


    def insert_into_persons(self, unique_id, person_name, person_surname, person_secondname, create_time, group_id, person_iin):
        print(unique_id, person_name, person_surname, person_secondname, group_id)
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # query
            insert_query = """INSERT INTO
                                fr.persons(
                                unique_id,
                                person_name,
                                person_surname,
                                person_secondname,
                                create_time,
                                group_id) VALUES (
                                %(uid)s,%(p_name)s,%(p_surname)s,%(p_sname)s,%(c_time)s,%(g_id)s,%(p_iin)s
                                )"""
            # execute the INSERT statement
            cur.execute(insert_query, {'uid': unique_id, 
                                        'p_name': person_name, 
                                        'p_surname': person_surname, 
                                        'p_sname': person_secondname, 
                                        'c_time': create_time, 
                                        'g_id': group_id, 
                                        'p_iin': person_iin})
            # commit the changes to the database
            conn.commit()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return False
        finally:
            if conn is not None:
                conn.close()
        return True


    def delete_from_persons(self, unique_id):
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host = self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # execute the INSERT statement
            cur.execute("DELETE FROM fr.persons WHERE unique_id = %s".format(unique_id))
            # commit the changes to the database
            conn.commit()
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return False
        finally:
            if conn is not None:
                conn.close()
        return True


    def search_from_gbdfl_faiss_top_n(self, faiss_index, one_vector, top_n):
        try:
            topn = 1
            if faiss_index.ntotal >= top_n:
                topn = top_n
            else:
                topn = faiss_index.ntotal
            nprb = 4096
            if faiss_index.ntotal > 1000000:
                faiss_index.nprobe = nprb
            else:
                faiss_index.nprobe = 256
            # print('ntotal:', faiss_index.ntotal)
            query = np.array([one_vector], dtype=np.float32)
            D, I = faiss_index.search(query, topn)

            return D, I
        except:
            return None, None


    def get_blob_info_from_database(self, ids):
        # Search from big database according to given red_id from faiss index with face vectors
        conn = None
        iin = None
        #print(ids)
        try:
            # connect to the PostgresQL database
            conn = psycopg2.connect(host=self.host, port=self.port, dbname=self.dbname, user=self.user, password=self.pwd)
            # create a new cursor object
            cur = conn.cursor()
            # execute the INSERT statement
            if len(ids) == 1:
                sql_str = "SELECT ud_code,gr_code,lastname,firstname,secondname FROM fr.unique_ud_gr WHERE ud_code = '{ids}'".format(ids=ids[0])
            else:
                sql_str = "SELECT ud_code,gr_code,lastname,firstname,secondname FROM fr.unique_ud_gr WHERE ud_code in {ids}".format(ids=ids)
            print(sql_str)
            #print(sql_str)
            cur.execute(sql_str)
            # commit the changes to the database
            blob = cur.fetchall()
            print('database blob:', blob)
            # close the communication with the PostgresQL database
            cur.close()
        except Exception as error:
            print('Error: ' + str(error))
            return None
        finally:
            if conn is not None:
                conn.close()
        return blob