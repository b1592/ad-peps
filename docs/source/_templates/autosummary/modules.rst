{{ fullname | escape | underline }}

Description
-----------

.. currentmodule:: {{ fullname | escape }}

.. automodule:: {{ fullname | escape }}

  {% if attributes %}
  Attributes
  -------

  .. autosummary::
    :toctree: generated

    {% for attribute in attributes %}
      {{ attribute }}
    {% endfor %}

  {% endif %}

  {% if modules %}
  Modules
  -------

  .. autosummary::
    :toctree: generated

    {% for module in modules %}
      {{ module }}
    {% endfor %}

  {% endif %}

  {% if classes %}
  Classes
  -------
  .. autosummary::
      :toctree: generated

      {% for class in classes %}
          {{ class }}
      {% endfor %}

  {% endif %}

  {% if functions %}
  Functions
  ---------
  .. autosummary::
      :toctree: generated

      {% for function in functions %}
          {{ function }}
      {% endfor %}

  {% endif %}
