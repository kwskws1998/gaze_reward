import lmdb
import pickle
import csv
import sys
import torch
import pathlib


class LMDBStorage:
    def __init__(self, db_path="buffer_train.lmdb", map_size=1024 * 1024 * 1024 * 20):
        self.db_path = db_path

        self.map_size = map_size
        self.env = lmdb.open(db_path, map_size=map_size)
        # with self.env.begin(write=True) as txn:
        # self.count = pickle.loads(txn.get(b'count', pickle.dumps(0)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def add(self, key, item, use_pickle=True):
        with self.env.begin(write=True) as txn:
            key = f"{key}".encode("ascii")
            if use_pickle:
                txn.put(key, pickle.dumps(item))
            else:
                txn.put(key, item)
            # self.count += 1
            # txn.put(b'count', pickle.dumps(self.count))

    def getItem(self, key, use_pickle=True):
        with self.env.begin(write=False) as txn:
            key = f"{key}".encode("ascii")
            item = txn.get(key)
            if item is not None:
                if use_pickle:
                    return pickle.loads(item)
                else:
                    return item
            else:
                # Key doesn't exist; handle the case here
                return None

    # def __len__(self):
    #     return self.count

    def close(self):
        self.env.close()

    def __getstate__(self):
        # Return the state without the env (since it's not picklable)
        state = self.__dict__.copy()
        state["env"] = None  # Exclude the lmdb environment from being pickled
        return state

    def __setstate__(self, state):
        # Restore the state and re-open the lmdb environment
        self.__dict__.update(state)
        self.env = lmdb.open(self.db_path, map_size=self.map_size)

    def all_items(self, use_pickle=True):
        """
        A generator function to iterate over all keys and values in the LMDB database.

        Yields:
            tuple: A tuple containing the key (as a string) and the associated data (unpickled object).
        """
        with self.env.begin(write=False) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if use_pickle:
                    yield key.decode("ascii"), pickle.loads(value)
                else:
                    yield key.decode("ascii"), value

    def save_to_csv(self, csv_file_path):
        """
        Saves all key-value pairs from the LMDB database to a CSV file.

        Args:
            csv_file_path (str): The path where the CSV file will be saved.
        """
        # find out keys in the stored cache, but just the first item
        for key, value in self.all_items():
            return_dict = {"key": key}  # Ensure 'key' is the first column
            for k in value.keys():
                return_dict[k] = value[k].numpy().tolist()  # Convert tensor to list

            headers = ["key"] + list(value.keys())
            break

        with open(csv_file_path, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()  # Write header

            for key, value in self.all_items():
                return_dict = {"key": key}  # Ensure 'key' is the first column
                for k in value.keys():
                    return_dict[k] = (
                        value[k].detach().numpy().tolist()
                    )  # Convert tensor to list
                    # print(value[k].dtype, value[k].shape)
                writer.writerow(return_dict)  # Store as plain text

    def load_from_csv(self, csv_file_path):
        """
        Loads key-value pairs from a CSV file into the LMDB database.
        The "key" column is used as the key, and the remaining columns are loaded as a dictionary.
        Each item in the dictionary is converted into a torch.tensor.

        Args:
            csv_file_path (str): The path to the CSV file to be loaded.
        """

        # Increase the field size limit to avoid the field size error
        csv.field_size_limit(sys.maxsize)
        with open(csv_file_path, mode="r", newline="") as file:
            reader = csv.DictReader(file)

            for row in reader:
                key = row.pop("key")  # Extract the 'key' column
                value_dict = {}

                # Convert the remaining items to torch tensors
                for k, v in row.items():
                    # Convert the string representation back into a list and then to a tensor
                    value_dict[k] = torch.tensor(eval(v))

                # Add the key and its corresponding value_dict back into the database
                # print(value_dict)
                self.add(key, value_dict)
                # for k in value_dict.keys():
                # print(value_dict[k].dtype, value_dict[k].shape)


# # Example usage
# with LMDBMemoryBuffer(buffer_size=1000, batch_size=10, seed=42) as buffer:
#     buffer.add(your_item)
#     sampled_items = buffer.sample(5)
#     # The LMDB environment will be automatically closed when exiting the with block
import argparse

if __name__ == "__main__":
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser(
        description="LMDB Storage script to export or import CSV data."
    )

    # Add optional arguments for db_path, csv_file_path, and the action (import/export)
    parser.add_argument(
        "--action",
        type=str,
        choices=["export", "import", "copy"],
        required=True,
        help="Specify whether to export the database to CSV or import from CSV.",
    )
    parser.add_argument(
        "--to_db_path", type=str, required=False, help="Path to the LMDB database."
    )
    parser.add_argument(
        "--from_db_path",
        type=str,
        required=False,
        help="Path to the LMDB database from where to copy data.",
    )
    parser.add_argument(
        "--csv_file_path",
        type=str,
        required=False,
        help="Path to the CSV file for importing/exporting.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug mode (default: False)."
    )

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Create an instance of LMDBStorage with the provided db_path

    if args.action == "export":
        with LMDBStorage(db_path=args.from_db_path) as storage:
            print(f"Exporting data from {args.from_db_path} to {args.csv_file_path}...")
            storage.save_to_csv(args.csv_file_path)
            print("Export completed.")
    elif args.action == "import":
        with LMDBStorage(db_path=args.to_db_path) as storage:
            print(f"Importing data from {args.csv_file_path} into {args.to_db_path}...")
            storage.load_from_csv(args.csv_file_path)
            print("Import completed.")
    elif args.action == "copy":
        if args.from_db_path is None:
            print("argument --from_db_path needed when copy")
            sys.exit()

        if args.to_db_path is None:
            print("argument --to_db_path needed when copy")
            sys.exit()

        with LMDBStorage(db_path=args.to_db_path) as storage:
            with LMDBStorage(db_path=args.from_db_path) as orig_storage:
                for item in orig_storage.all_items(use_pickle=False):
                    key, value = item
                    ##what if that key already exists? check consistency:
                    dest_value = storage.getItem(key, use_pickle=False)
                    if dest_value is not None and dest_value != value:
                        print(
                            "Key",
                            key,
                            "exists in target buffer",
                            args.to_db_path,
                            "but the stored data is different from the origin",
                        )
                        dict_value = pickle.loads(value)
                        dict_dest_value = pickle.loads(dest_value)
                        same_value = True
                        for k in dict_value:
                            if (
                                same_value
                                and dict_value[k].detach().numpy().tolist()
                                != dict_dest_value[k].detach().numpy().tolist()
                            ):
                                same_value = False

                            if args.debug:
                                print(
                                    k,
                                    "LIST orig_data == dest_data",
                                    dict_value[k].detach().numpy().tolist()
                                    == dict_dest_value[k].detach().numpy().tolist(),
                                )
                                print(
                                    k,
                                    "SHAPES orig_data,  dest_data",
                                    dict_value[k].shape,
                                    dict_dest_value[k].shape,
                                )
                                print(
                                    k,
                                    "REQUIRES_GRAD orig_data,  dest_data",
                                    dict_value[k].requires_grad,
                                    dict_dest_value[k].requires_grad,
                                )
                                if dict_value[k].shape == dict_dest_value[k].shape:
                                    print(
                                        k,
                                        "TORCH DETACH orig_data == dest_data",
                                        torch.allclose(
                                            dict_value[k].detach(),
                                            dict_dest_value[k].detach(),
                                        ),
                                    )
                                    print(
                                        k,
                                        "TORCH orig_data == dest_data",
                                        torch.allclose(
                                            dict_value[k], dict_dest_value[k]
                                        ),
                                    )
                        if same_value:
                            print(
                                "The pickled data seems to be different but the internal values are actually the same"
                            )
                        else:
                            print("Both pickle and internal data are different")
                    storage.add(key, value, use_pickle=False)
