from src.representations.time_surface import ToTimesurface  # Assuming the class is saved in to_timesurface.py
from utils import generate_synthetic_events, read_large_file_pandas
import numpy as np
from tqdm import tqdm 



def main():
    sensor_size = (128, 128, 2)  # Example sensor size
    surface_dimensions = (5, 5)  # Example surface dimensions
    tau = 5e3
    decay = "lin"

    # Instantiate ToTimesurface
    timesurface_generator = ToTimesurface(sensor_size, surface_dimensions, tau, decay)

    # Generate synthetic events
    num_events = 1000
    events = generate_synthetic_events(num_events, sensor_size)
    indices = np.arange(num_events)  # Example indices

    # Generate time surfaces
    time_surfaces = timesurface_generator(events, indices)

    # Basic output verification
    print("Generated time surfaces shape:", time_surfaces.shape)
    print("First time surface:", time_surfaces[0, :, :, :])





def second():  
    # Load data from txt

    # Example usage
    file_path = 'datasets/uzh_fpv/Tracks/Track1/txt/events.txt'
    df=read_large_file_pandas(file_path, nrows=10, chunksz = 100000)

    print(df.columns)
    event_batch=df
    print(event_batch)

    sensor_size = (128, 128, 2)  # Example sensor size
    surface_dimensions = (5, 5)  # Example surface dimensions
    tau = 5e3
    decay = "lin"

        # Instantiate ToTimesurface
    timesurface_generator = ToTimesurface(sensor_size, surface_dimensions, tau, decay)

    #events = generate_synthetic_events(10, sensor_size)
    dict_events={}
    #indices=event_batch[:,0]
    print(f"t {event_batch['t']}")
    print(f"x col value {event_batch['x'].to_numpy()}")
    #indices = np.arange(events.size())  # Example indices 

    # Convert DataFrame to dictionary of lists
    dict_of_lists = df.to_dict(orient='list')
    print(f"dict of lists: {dict_of_lists}")
    # Convert lists to NumPy arrays
    dict_of_arrays = {key: np.array(value) for key, value in dict_of_lists.items()}

    print(f"dict of arrays: {dict_of_arrays}")

    # Generate time surfaces
    time_surfaces = timesurface_generator(dict_of_arrays, event_batch.index.to_numpy())

    # Basic output verification
    print("Generated time surfaces shape:", time_surfaces.shape)
    print("First time surface:", time_surfaces[0, :, :, :])

    #   method="sae" #SAE
    #   ev_processor=EventProcessor(method=method)

    #   result=ev_processor.process_batch(event_batch)
    # print(f"result: {result}")

if __name__ == "__main__":
    main()
    second()



