from pyrep import PyRep

pr = PyRep()
# Launch the application with a scene file in headless mode
pr.launch(headless=False) 
pr.start()  # Start the simulation

# Do some stuff
for _ in range(1000):
    pr.step()

pr.stop()  # Stop the simulation
pr.shutdown()  # Close the application