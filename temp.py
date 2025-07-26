# conn, cur = init_db(db_path)
# with Bert2VecModel(source_path=config().dest_path, in_mem=False, new_model=True) as dest_model:
#     offset = 0
#     while True:
#         cur.execute(
#             f"""
#                        SELECT {config().input_column}, {config().entries_column} FROM my_table
#                        LIMIT ? OFFSET ?
#                        """,
#             (config().chunk_size, offset),
#         )
#         rows = cur.fetchall()
#         if not rows:
#             break
#
#         for input_text, pickled_blob in rows:
#             obj = pickle.loads(pickled_blob)
#             yield input_text, obj

# print(f"Starting...")
# start_time = epoch_time = time()
# for idx, sentence in enumerate(corpus):
#     sentence_entries = unite_sentence_tokens(sentence=sentence, bert2vec_model=bert2vec_model)
#
#     for entry in sentence_entries:
#         dest_model.add_entry(entry)
#     if idx%config().print_checkpoint_count == 0:
#         end_time = time()
#         run_time = end_time - epoch_time
#         total_time = end_time - start_time
#         average_time = total_time/(idx+1)
#         time_remaining = (sentence_count - idx + 1)*average_time
#         print(f"""Finished {idx+1:,}/{sentence_count:,} sentences in {run_time:.4f} seconds\n"""
#               f"""Total time: {timedelta(seconds=total_time)}\n"""
#               f"""Average time: {average_time} seconds/sentence,\n"""
#               f"""Time remaining: {timedelta(seconds=time_remaining)}\n"""
#               f"""Last saved checkpoint: {last_save_checkpoint:,} sentences""")
#         epoch_time = end_time
#     if idx%config().save_checkpoint_count == 0:
#         last_save_checkpoint = idx
#         dest_model.save_data()
# print(f"Done {sentence_count} sentences")

# source_path = config().bert2vec_path
# source_path = "shared_files/shelve-unite/shelve.slv"
# dest_path = "data/shelve-unite/shelve.slv"
# convert_to_pydantic(source_path=source_path, dest_path=dest_path)
