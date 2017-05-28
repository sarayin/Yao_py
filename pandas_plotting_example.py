# We can ask for ALL THE AXES and put them into axes
fig, axes = plt.subplots(nrows=6, ncols=8, sharex=True, sharey=True, figsize=(18,10))
axes_list = [item for sublist in axes for item in sublist]

ordered_country_names = grouped['GDP_per_capita'].last().sort_values(ascending=False).index

# Now instead of looping through the groupby
# you CREATE the groupby
# you LOOP through the ordered names
# and you use .get_group to get the right group
grouped = df.head(3000).groupby("Country")

first_year = df['Year'].min()
last_year = df['Year'].max()

for countryname in ordered_country_names:
    selection = grouped.get_group(countryname)

    ax = axes_list.pop(0)
    selection.plot(x='Year', y='GDP_per_capita', label=countryname, ax=ax, legend=False)
    ax.set_title(countryname)
    ax.tick_params(
        which='both',
        bottom='off',
        left='off',
        right='off',
        top='off'
    )
    ax.grid(linewidth=0.25)
    ax.set_xlim((first_year, last_year))
    ax.set_xlabel("")
    ax.set_xticks((first_year, last_year))
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    max_year = selection['Year'].max()
    gdp_value = float(selection[df['Year'] == max_year]['GDP_per_capita'])
    ax.set_ylim((0, 100000))
    ax.scatter(x=[max_year], y=[gdp_value], s=70, clip_on=False, linewidth=0)
    ax.annotate(str(int(gdp_value / 1000)) + "k", xy=[max_year, gdp_value], xytext=[7, -2], textcoords='offset points')

# Now use the matplotlib .remove() method to
# delete anything we didn't use
for ax in axes_list:
    ax.remove()

plt.tight_layout()
plt.subplots_adjust(hspace=1)
