{{ fullname | escape | underline }}

.. currentmodule:: {{ fullname | escape }}

.. automodule:: {{ fullname }}
  :members:

{% if modules %}
Modules
-------

.. autosummary::
  :toctree:

  {% for module in modules %}
    {{ module }}
  {% endfor %}

{% else %}

{% if classes %}
Classes
-------
{% for class in classes %}
{{ class | escape | underline }}
  .. autoclass:: {{ class }}
    :members:
    :member-order: bysource
{% endfor %}
{% endif %}

{% if attributes %}
Attributes
-----------

{% for attribute in attributes %}
.. autoattribute:: {{ attribute }}
{% endfor %}

{% endif %}

{% if functions %}
Functions
---------

.. .. automodule:: {{ fullname}}
..   :members:

{% for function in functions %}
.. autofunction:: {{ function }}

.. {% endfor %}

{% endif %}


{% endif %}

