
import streamlit as st
import pandas as pd
import allocator
import copy

# Initialize Page
st.set_page_config(page_title="PC Component Configurator", layout="wide")

st.title("🖥️ PC Component Configurator")
st.markdown("Build your dream PC based on your budget and needs.")

# Sidebar for Inputs
with st.sidebar:
    st.header("Configuration")
    
    budget = st.number_input("Enter Budget ($)", min_value=450, max_value=10000, value=1200, step=50)
    
    use_case = st.selectbox(
        "Select Use Case",
        options=['general', 'gaming', 'workstation', 'server', 'custom'],
        index=1,
        format_func=lambda x: x.capitalize()
    )
    
    # Description
    descriptions = {
        'general': 'Suitable for everyday tasks and general usage.',
        'gaming': 'Optimized for high performance gaming experiences.',
        'workstation': 'Designed for professional work and high productivity.',
        'server': 'Ideal for server applications and hosting services.',
        'custom': 'Customize each component to fit your specific needs.'
    }
    st.info(descriptions[use_case])

# Manage Preferences
# We work with a copy to avoid mutating the global preset permanently if we were mocking it, 
# but here we just read it.
selected_preference = copy.deepcopy(allocator.preference_preset[use_case])

if use_case == 'custom':
    with st.expander("Advanced Allocation Settings"):
        st.write("Adjust Budget Allocation per Component:")
        alloc = selected_preference['allocation']
        
        # Create sliders for each component allocation
        new_alloc = {}
        for key, val in alloc.items():
            new_alloc[key] = st.slider(f"{key.upper()} Allocation", 0.0, 1.0, val, 0.05)
        
        # Normalize to ensure sum includes changes correctly (simple normalization)
        total = sum(new_alloc.values())
        if total > 0:
            for k in new_alloc:
                new_alloc[k] = new_alloc[k] / total
        
        selected_preference['allocation'] = new_alloc
        st.write(f"Total Allocation: {sum(new_alloc.values()):.2f}")


def generate_build():
    # RESET GLOBAL STATE in allocator module
    # This is necessary because allocator.py uses a global dictionary for constraints
    allocator.constraints = {
        'gpu': None,
        'wattage': 0,
        'socket': None,
        'supported_mem': None,
        'chipset': None,
        'form_factor': None,
        'max_memory': None,
        'memory_slots': None,
        'M.2': 0,
        'SATA': 0,
    }

    try:
        with st.spinner('Generating optimal build...'):
            part_list, constraints, leftover = allocator.allocator(budget, selected_preference)
            return part_list, constraints, leftover
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None, None

if st.button("Generate Build", type="primary"):
    part_list, constraints, leftover = generate_build()
    
    if part_list:
        st.success(f"Build Generated! Leftover Budget: ${leftover}")
        
        # Display Stats
        st.subheader("Build Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Est. Wattage", f"{constraints.get('wattage', 0) * 1.5:.0f}W")
        c2.metric("Socket", constraints.get('socket', 'N/A'))
        c3.metric("Form Factor", constraints.get('form_factor', 'N/A'))
        c4.metric("GPU", "Included" if constraints.get('gpu') else "Integrated/None")

        st.markdown("---")
        st.subheader("Component List")

        # Helper to display component
        def display_component(title, component):
            if not component:
                st.warning(f"No {title} found within budget.")
                return

            with st.container():
                st.markdown(f"### {title}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"**{component.get('name', 'Unknown')}**")
                    # Description details
                    desc = component.get('description', {})
                    if isinstance(desc, dict):
                        st.caption(" | ".join([f"{k}: {v}" for k, v in desc.items()]))
                    else:
                        st.caption(str(desc))
                with col2:
                    st.write(f"**${component.get('price', 0)}**")
                    st.metric("Score", f"{component.get('score', 0):.1f}")
                st.divider()

        # Display Main Components
        display_component("CPU", part_list.get('cpu'))
        display_component("Motherboard", part_list.get('motherboard'))
        display_component("Memory (RAM)", part_list.get('ram'))
        display_component("Graphics Card (GPU)", part_list.get('gpu'))
        
        # Storage is a list
        storage_list = part_list.get('storage', [])
        if storage_list:
            for i, sto in enumerate(storage_list):
                display_component(f"Storage {i+1}", sto)
        else:
            st.warning("No Storage found within budget.")

        display_component("Power Supply (PSU)", part_list.get('psu'))

